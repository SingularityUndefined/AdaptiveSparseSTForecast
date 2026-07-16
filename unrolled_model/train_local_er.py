#!/usr/bin/env python

import argparse
import os
import random
import socket
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from dataset import LocalERGraphSignalDataset, make_local_er_split_loaders
from generate_dataset import generate_local_er_dataset
from model import UnrolledGEM


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a local ER dataset, split it, train UnrolledGEM, and run validation/test."
    )
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--reuse_dataset", type=int, choices=[0, 1], default=0)

    parser.add_argument("--num_nodes", type=int, default=100)
    parser.add_argument("--n_data", type=int, default=50)
    parser.add_argument("--k0", type=int, default=10)
    parser.add_argument("--k1", type=int, default=4)
    parser.add_argument("--dataset_size", type=int, default=120)
    parser.add_argument("--diag_shift", type=float, default=0.1)
    parser.add_argument("--noise_std", type=float, default=0.25)
    parser.add_argument("--weight_low", type=float, default=0.5)
    parser.add_argument("--weight_high", type=float, default=1.5)
    parser.add_argument("--same_graph", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--sample_mode",
        type=str,
        choices=["fixed_k1", "er"],
        default="fixed_k1",
    )

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--E_iters", type=int, default=4)
    parser.add_argument("--M_iters", type=int, default=4)
    parser.add_argument("--GD_step_init", type=float, default=0.1)
    parser.add_argument("--mu_init", type=float, default=0.2)
    parser.add_argument("--gamma_init", type=float, default=0.4)
    parser.add_argument("--c", type=float, default=20.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--alpha_init", type=float, default=0.5)
    parser.add_argument(
        "--er_backend",
        type=str,
        choices=["sksparse", "torch_dense"],
        default="torch_dense",
    )
    parser.add_argument("--er_backward_chunk_size", type=int, default=1024)
    parser.add_argument("--recon_loss_weight", type=float, default=1.0)
    parser.add_argument("--graph_loss_weight", type=float, default=0.05)
    parser.add_argument(
        "--init_graph_mode",
        type=str,
        choices=["candidate", "truth"],
        default="candidate",
    )

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _default_dataset_path(args) -> str:
    filename = (
        f"n_{args.num_nodes}_ndata_{args.n_data}_k0_{args.k0}_k1_{args.k1}_"
        f"mode_{args.sample_mode}_noise_{args.noise_std}_size_{args.dataset_size}_"
        f"seed_{args.seed}.pt"
    )
    return os.path.join("unrolled_model", "generated_data", filename)


def _generate_dataset_file(args) -> str:
    dataset_path = args.dataset_path or _default_dataset_path(args)
    if args.reuse_dataset and os.path.exists(dataset_path):
        print(f"Reusing dataset: {dataset_path}")
        return dataset_path

    dataset = generate_local_er_dataset(
        num_nodes=args.num_nodes,
        n_data=args.n_data,
        k0=args.k0,
        k1=args.k1,
        dataset_size=args.dataset_size,
        diag_shift=args.diag_shift,
        noise_std=args.noise_std,
        weight_low=args.weight_low,
        weight_high=args.weight_high,
        as_tensor=True,
        random_state=args.seed,
        same_graph=bool(args.same_graph),
        sample_mode=args.sample_mode,
    )
    output_dir = os.path.dirname(dataset_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(dataset, dataset_path)
    print(f"Saved dataset to {dataset_path}")
    return dataset_path


def _resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _expand_graph_tensor(graph: torch.Tensor, num_heads: int, device: torch.device) -> torch.Tensor:
    graph = graph.to(device)
    if graph.dim() != 4:
        raise ValueError("graph tensor must have shape (B, 1_or_H, N, k).")
    if graph.size(1) == num_heads:
        return graph
    if graph.size(1) != 1:
        raise ValueError(
            f"graph tensor head dimension must be 1 or {num_heads}, got {graph.size(1)}."
        )
    return graph.expand(-1, num_heads, -1, -1)


def _build_initial_graph(batch: Dict[str, torch.Tensor], num_heads: int, device: torch.device, mode: str) -> torch.Tensor:
    if mode == "truth":
        return _expand_graph_tensor(batch["W_true"].float(), num_heads, device)
    candidate_mask = _expand_graph_tensor(batch["candidate_mask"].float(), num_heads, device)
    return candidate_mask


def _compute_batch_metrics(model, batch, args, device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    y = batch["y"].to(device).float()
    x_clean = batch["x_clean"].to(device).float()
    W_true = _expand_graph_tensor(batch["W_true"].float(), args.num_heads, device)
    S_true = _expand_graph_tensor(batch["S_true"].bool(), args.num_heads, device)
    candidate_mask = _expand_graph_tensor(batch["candidate_mask"].bool(), args.num_heads, device)

    W_o = _build_initial_graph(batch, args.num_heads, device, args.init_graph_mode)
    x_pred, W_pred, S_pred = model(y, W_o)

    x_pred_mean = x_pred.mean(dim=2)
    recon_loss = F.mse_loss(x_pred_mean, x_clean)
    graph_loss = F.mse_loss(W_pred, W_true)
    loss = args.recon_loss_weight * recon_loss + args.graph_loss_weight * graph_loss

    edge_mask = candidate_mask
    pred_mask = S_pred.bool() & edge_mask
    true_mask = S_true & edge_mask
    tp = (pred_mask & true_mask).sum().item()
    fp = (pred_mask & ~true_mask).sum().item()
    fn = ((~pred_mask) & true_mask).sum().item()
    tn = ((~pred_mask) & (~true_mask) & edge_mask).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    edge_f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    edge_accuracy = (tp + tn) / max(int(edge_mask.sum().item()), 1)

    metrics = {
        "loss": float(loss.detach().item()),
        "recon_loss": float(recon_loss.detach().item()),
        "graph_loss": float(graph_loss.detach().item()),
        "edge_precision": float(precision),
        "edge_recall": float(recall),
        "edge_f1": float(edge_f1),
        "edge_accuracy": float(edge_accuracy),
    }
    return loss, metrics


def _run_epoch(model, loader, optimizer, args, device: torch.device, train: bool) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    totals: Dict[str, float] = {}
    total_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch_size = int(batch["y"].size(0))
            if train:
                optimizer.zero_grad(set_to_none=True)
            loss, metrics = _compute_batch_metrics(model, batch, args, device)
            if train:
                loss.backward()
                optimizer.step()

            total_samples += batch_size
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size

    return {key: value / max(total_samples, 1) for key, value in totals.items()}


def _build_model(sample: Dict[str, torch.Tensor], args, device: torch.device) -> UnrolledGEM:
    neighbor_list = sample["neighbor_list"].long().to(device)
    candidate_mask = sample["candidate_mask"].bool().to(device)
    num_nodes = int(neighbor_list.size(0))
    model = UnrolledGEM(
        num_nodes=num_nodes,
        neighbor_list=neighbor_list,
        input_neighbor_mask=candidate_mask,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        E_iters=args.E_iters,
        M_iters=args.M_iters,
        GD_step_init=args.GD_step_init,
        mu_init=args.mu_init,
        gamma_init=args.gamma_init,
        c=args.c,
        scale=True,
        epsilon=args.epsilon,
        alpha_init=args.alpha_init,
        er_backend=args.er_backend,
        er_backward_chunk_size=args.er_backward_chunk_size,
    )
    return model.to(device)


def _format_metrics(name: str, metrics: Dict[str, float]) -> str:
    return (
        f"{name}: "
        f"loss={metrics['loss']:.6f}, "
        f"recon={metrics['recon_loss']:.6f}, "
        f"graph={metrics['graph_loss']:.6f}, "
        f"edge_f1={metrics['edge_f1']:.4f}, "
        f"edge_acc={metrics['edge_accuracy']:.4f}"
    )


def main():
    print(socket.gethostname())
    args = _parse_args()
    _set_seed(args.seed)
    device = _resolve_device(args.device)

    dataset_path = _generate_dataset_file(args)
    dataset = LocalERGraphSignalDataset(dataset_path, map_location="cpu")
    loaders, splits = make_local_er_split_loaders(
        dataset_path,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        num_workers=args.num_workers,
        map_location="cpu",
    )
    sample = dataset[0]
    model = _build_model(sample, args, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("device:", device)
    print("dataset path:", dataset_path)
    print("split sizes:", {name: len(split) for name, split in splits.items()})
    print(
        "model config:",
        {
            "num_heads": args.num_heads,
            "num_blocks": args.num_blocks,
            "E_iters": args.E_iters,
            "M_iters": args.M_iters,
            "er_backend": args.er_backend,
        },
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, loaders["train"], optimizer, args, device, train=True)
        val_metrics = _run_epoch(model, loaders["val"], optimizer, args, device, train=False)
        print(
            f"epoch {epoch:03d} | "
            f"{_format_metrics('train', train_metrics)} | "
            f"{_format_metrics('val', val_metrics)}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            if args.checkpoint_path:
                checkpoint_dir = os.path.dirname(args.checkpoint_path)
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(best_state, args.checkpoint_path)
                print(f"Saved checkpoint to {args.checkpoint_path}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _run_epoch(model, loaders["test"], optimizer, args, device, train=False)
    print(_format_metrics("test", test_metrics))


if __name__ == "__main__":
    main()
