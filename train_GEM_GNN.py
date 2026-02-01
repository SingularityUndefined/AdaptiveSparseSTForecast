from dataloader.data_utils import generate_y_from_grid, generate_kNN_from_grid
from model.utils import draw_graph
import torch
from torch.utils.data import DataLoader
from model.GNN_GEM_lightning import GEM_GNN
import lightning as L

class DummyGraphDataset(torch.utils.data.Dataset):
    def __init__(self, num_row, data_len, sigma=0.4):
        self.num_nodes = num_row * num_row
        self.sigma = sigma
        self.data_len = data_len
        _, self.data = generate_y_from_grid(num_row, sigma, data_len)
        # self.data = self.data.unsqueeze(0)  # (1, data_len, N)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.data[idx]


# =====================================
# 训练代码
# =====================================
if __name__ == "__main__":
    # 参数
    in_dim = 3
    hidden_dim = 16
    out_dim = 2
    lr = 1e-2
    batch_size = 32
    max_epochs = 13

    # 数据
    dataset = DummyGraphDataset(num_row=32, data_len=512, sigma=0.4)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 模型
    model = GEM_GNN(
        num_nodes=32*32,
        num_neighbors=48,
        neighbor_list=generate_kNN_from_grid(32, kernel=7, k=48),
        mu=0.2,
        gamma=0.4,
        emb_dim=8,
        feature_dim=in_dim,
        c=80,
        theta=0.5,
        method='CG',
        inv_method='L+J',
        CG_iters=5,
        PGD_iters=8,
        PGD_step_size=0.01,
        scale=True,
        GEM_iters=5,
        lr=lr
    )

    # Trainer
    trainer = L.Trainer(max_epochs=max_epochs, accelerator="cuda" if torch.cuda.is_available() else "cpu", devices=1, val_check_interval=4)
    trainer.fit(model, train_loader, DataLoader([torch.zeros(1)]))
    learned_graph = model.graph_list[-1]
    # # plot graph
    # draw_graph(
    #     model.neighbor_list,
    #     torch.tensor(learned_graph, device=model.neighbor_list.device),
    #     n_row=32,
    #     title="Learned Graph after Training",
    #     filename="learned_graph_after_training.png"
    # )