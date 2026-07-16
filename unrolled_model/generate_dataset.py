import argparse
import os
import socket

import numpy as np
import torch


def build_ring_neighbor_list(num_nodes, k0):
    """Build a symmetric local candidate list on a ring.

    Each node uses the k0 closest nodes on a circular ordering as possible
    neighbors: k0 / 2 on the left and k0 / 2 on the right.
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    if k0 <= 0:
        raise ValueError("k0 must be positive.")
    if k0 >= num_nodes:
        raise ValueError("k0 must be smaller than num_nodes.")
    if k0 % 2 != 0:
        raise ValueError("k0 must be even for a symmetric ring neighbor list.")

    radius = k0 // 2
    neighbor_list = np.empty((num_nodes, k0), dtype=np.int64)
    for node in range(num_nodes):
        neighbors = []
        for offset in range(1, radius + 1):
            neighbors.append((node - offset) % num_nodes)
            neighbors.append((node + offset) % num_nodes)
        neighbor_list[node] = np.asarray(neighbors, dtype=np.int64)
    return neighbor_list


def _neighbor_positions(neighbor_list):
    positions = {}
    num_nodes, k0 = neighbor_list.shape
    for node in range(num_nodes):
        for slot in range(k0):
            positions[(node, int(neighbor_list[node, slot]))] = slot
    return positions


def _sample_true_mask(neighbor_list, k1, rng, sample_mode):
    num_nodes, k0 = neighbor_list.shape
    if k1 < 0 or k1 > k0:
        raise ValueError("k1 must satisfy 0 <= k1 <= k0.")
    if sample_mode not in ("fixed_k1", "er"):
        raise ValueError("sample_mode must be 'fixed_k1' or 'er'.")

    positions = _neighbor_positions(neighbor_list)
    directed_mask = np.zeros((num_nodes, k0), dtype=bool)
    true_mask = np.zeros((num_nodes, k0), dtype=bool)

    if sample_mode == "fixed_k1":
        for node in range(num_nodes):
            selected_slots = rng.choice(k0, size=k1, replace=False)
            directed_mask[node, selected_slots] = True
            for slot in selected_slots:
                neighbor = int(neighbor_list[node, slot])
                reverse_slot = positions[(neighbor, node)]
                true_mask[node, slot] = True
                true_mask[neighbor, reverse_slot] = True
        return true_mask, directed_mask

    edge_probability = k1 / k0 if k0 else 0.0
    for node in range(num_nodes):
        for slot in range(k0):
            neighbor = int(neighbor_list[node, slot])
            if node < neighbor and rng.rand() < edge_probability:
                reverse_slot = positions[(neighbor, node)]
                true_mask[node, slot] = True
                true_mask[neighbor, reverse_slot] = True

    return true_mask, directed_mask


def _sample_symmetric_weights(neighbor_list, true_mask, rng, weight_low, weight_high):
    if weight_low <= 0.0 or weight_high <= 0.0 or weight_low > weight_high:
        raise ValueError("weights must satisfy 0 < weight_low <= weight_high.")

    num_nodes, k0 = neighbor_list.shape
    positions = _neighbor_positions(neighbor_list)
    W = np.zeros((num_nodes, k0), dtype=np.float64)

    for node in range(num_nodes):
        for slot in range(k0):
            neighbor = int(neighbor_list[node, slot])
            if node < neighbor and true_mask[node, slot]:
                reverse_slot = positions[(neighbor, node)]
                weight = rng.uniform(weight_low, weight_high)
                W[node, slot] = weight
                W[neighbor, reverse_slot] = weight

    return W


def _neighbor_weights_to_adjacency(neighbor_list, W):
    num_nodes, k0 = neighbor_list.shape
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for node in range(num_nodes):
        for slot in range(k0):
            neighbor = int(neighbor_list[node, slot])
            adjacency[node, neighbor] = W[node, slot]
    return 0.5 * (adjacency + adjacency.T)


def _precision_from_weights(neighbor_list, W, diag_shift):
    if diag_shift <= 0.0:
        raise ValueError("diag_shift must be positive.")

    adjacency = _neighbor_weights_to_adjacency(neighbor_list, W)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    theta_true = laplacian + diag_shift * np.eye(adjacency.shape[0])
    return theta_true, adjacency, laplacian


def _to_tensor_item(item):
    tensor_item = {}
    for key, value in item.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.bool_:
                tensor_item[key] = torch.from_numpy(value)
            elif np.issubdtype(value.dtype, np.integer):
                tensor_item[key] = torch.from_numpy(value).long()
            else:
                tensor_item[key] = torch.from_numpy(value).float()
        else:
            tensor_item[key] = value
    return tensor_item


def generate_local_er_dataset(
    num_nodes=100,
    n_data=50,
    k0=10,
    k1=4,
    dataset_size=1,
    diag_shift=1e-1,
    noise_std=0.25,
    weight_low=0.5,
    weight_high=1.5,
    as_tensor=True,
    random_state=0,
    same_graph=False,
    sample_mode="fixed_k1",
):
    """Generate Gaussian graph-signal data with a local random underlying graph.

    This mirrors the SpodNet data setting: build a true precision matrix
    ``Theta_true``, set ``Sigma_true = pinv(Theta_true)``, draw iid centered
    Gaussian samples, and compute their empirical covariance.  Here
    ``Theta_true`` is graph-structured: ``Theta_true = L(W_true) + diag_shift I``.
    Clean signals are sampled from ``N(0, Sigma_true)`` and observations add
    iid Gaussian noise: ``y_noisy = x_clean + noise_std * N(0, I)``.

    Public noisy samples are stored as ``y`` and ``y_noisy`` with shape
    ``(n_data, num_nodes)``; clean samples are stored as ``x_clean``.
    A training collate can stack them into ``(B, n_data, num_nodes)``; the
    unrolled model expands the head dimension internally.

    ``sample_mode='fixed_k1'`` matches the requested local selection rule:
    every node randomly chooses k1 of its k0 candidate neighbors, then the graph
    is symmetrized.  ``sample_mode='er'`` samples each undirected candidate edge
    independently with probability k1 / k0, so k1 is the expected degree.
    """
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative.")

    rng = np.random.RandomState(random_state)
    neighbor_list = build_ring_neighbor_list(num_nodes, k0)
    candidate_mask = np.ones((1, num_nodes, k0), dtype=bool)

    if same_graph:
        graph_rng = np.random.RandomState(random_state)
        true_mask, directed_true_mask = _sample_true_mask(
            neighbor_list,
            k1,
            graph_rng,
            sample_mode,
        )
        W_true = _sample_symmetric_weights(
            neighbor_list,
            true_mask,
            graph_rng,
            weight_low,
            weight_high,
        )

    dataset = []
    for _ in range(dataset_size):
        if not same_graph:
            true_mask, directed_true_mask = _sample_true_mask(
                neighbor_list,
                k1,
                rng,
                sample_mode,
            )
            W_true = _sample_symmetric_weights(
                neighbor_list,
                true_mask,
                rng,
                weight_low,
                weight_high,
            )

        theta_true, adjacency_true, laplacian_true = _precision_from_weights(
            neighbor_list,
            W_true,
            diag_shift,
        )
        sigma_true = np.linalg.pinv(theta_true)
        x_clean = rng.multivariate_normal(
            mean=np.zeros(num_nodes),
            cov=sigma_true,
            size=n_data,
        )
        noise = noise_std * rng.randn(n_data, num_nodes)
        y_noisy = x_clean + noise
        empirical_cov = np.cov(y_noisy, bias=True, rowvar=False)
        empirical_cov_clean = np.cov(x_clean, bias=True, rowvar=False)

        item = {
            "y": y_noisy,
            "y_noisy": y_noisy,
            "x_clean": x_clean,
            "noise": noise,
            "empirical_cov": empirical_cov,
            "empirical_cov_clean": empirical_cov_clean,
            "Theta_true": theta_true,
            "Sigma_true": sigma_true,
            "L_true": laplacian_true,
            "A_true": adjacency_true,
            "W_true": W_true[np.newaxis, ...],
            "S_true": true_mask[np.newaxis, ...],
            "directed_S_true": directed_true_mask[np.newaxis, ...],
            "neighbor_list": neighbor_list,
            "candidate_mask": candidate_mask,
            "k0": k0,
            "k1": k1,
            "edge_probability": k1 / k0,
            "diag_shift": diag_shift,
            "noise_std": noise_std,
            "sample_mode": sample_mode,
        }
        dataset.append(_to_tensor_item(item) if as_tensor else item)

    return dataset


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate local ER graph-signal datasets")
    parser.add_argument("--set_type", type=str, default="train")
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--n_data", type=int, required=True)
    parser.add_argument("--k0", type=int, required=True)
    parser.add_argument("--k1", type=int, required=True)
    parser.add_argument("--dataset_size", type=int, required=True)
    parser.add_argument("--diag_shift", type=float, default=1e-1)
    parser.add_argument("--noise_std", type=float, default=0.25)
    parser.add_argument("--weight_low", type=float, default=0.5)
    parser.add_argument("--weight_high", type=float, default=1.5)
    parser.add_argument("--as_tensor", type=int, choices=[0, 1], default=1)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--same_graph", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--sample_mode",
        type=str,
        choices=["fixed_k1", "er"],
        default="fixed_k1",
    )
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    print(socket.gethostname())
    args = _parse_args()
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
        as_tensor=bool(args.as_tensor),
        random_state=args.random_state,
        same_graph=bool(args.same_graph),
        sample_mode=args.sample_mode,
    )

    output = args.output
    if output is None:
        os.makedirs(args.set_type, exist_ok=True)
        output = (
            f"{args.set_type}/n_{args.num_nodes}_ndata_{args.n_data}_"
            f"k0_{args.k0}_k1_{args.k1}_mode_{args.sample_mode}_"
            f"noise_{args.noise_std}_size_{args.dataset_size}_"
            f"seed_{args.random_state}.pt"
        )
    else:
        output_dir = os.path.dirname(output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    torch.save(dataset, output)
    print(f"Saved dataset to {output}.")


if __name__ == "__main__":
    main()
