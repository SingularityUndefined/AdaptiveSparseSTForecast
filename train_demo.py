import torch
import torch.nn as nn
from model.unrolled_GEM import UnrolledGEM
from dataloader.data_utils import *
import time
from tqdm import tqdm   

# Generate graph and data
n_row = 32
kernel = 5
assert kernel % 2 == 1, "kernel size must be odd"
k = kernel ** 2 - 1
feature_dim = 6
num_nodes = n_row * n_row
# signal generation
x, y = generate_y_from_grid(n_row, sigma=0.4, n=512)

# base graph generation
nearest_neighbors = generate_kNN_from_grid(n_row, kernel, k)
neighbor_mask = nearest_neighbors != -1

# Model initialization
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model_param_dict = {
    'num_nodes': num_nodes,
    'num_neighbors': k,
    'neighbor_list': nearest_neighbors.to(device),
    'mu': 0.2,
    'gamma': 0.4,
    'emb_dim': 8,
    'feature_dim': feature_dim,
    'c': 16,
    'theta': 0.5,
    'method': 'CG',
    'inv_method': 'L+J',
    'E_step_iters': 3,
    'inv_CG_iters': 4,
    'PGD_iters': 5,
    'PGD_step_size': 0.02,
    'M1_step_size': 0.02,
    'scale': True,
    'GEM_iters': 3,
    'full_unrolling': True
}

model = UnrolledGEM(**model_param_dict).to(device)
# model size
total_params = sum(p.numel() for p in model.parameters())
print(f'Model total parameters: {total_params}')

# training demo, compute memory cost and FLOPS
model.train()
y = y.to(device)
target = torch.zeros_like(y).to(device)
batch_size = 32
num_batches = y.shape[0] // batch_size


optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# device = torch.device("cuda")

model = model.to(device)
start_time = time.time()
for epochs in tqdm(range(5)):
    for i in range(num_batches):
        # optimizer.zero_grad()
        # torch.cuda.reset_peak_memory_stats(device)
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize(device)  # 👈 必须

        y_batch = y[i*batch_size:(i+1)*batch_size].to(device)

        # import torch.profiler

        # with torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True
        # ) as prof:
        #     output, _, _, _ = model(y_batch)
        #     target = torch.zeros_like(output)
        #     loss = nn.MSELoss()(output, target)
        #     loss.backward()
        #     optimizer.step()

        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        output, _, _, _ = model(y_batch)
        target = x[i*batch_size:(i+1)*batch_size].to(device)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()


        # torch.cuda.synchronize(device)  # 👈 必须

        # max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        # print(f"Batch {i+1}/{num_batches}, Max GPU memory allocated: {max_memory:.2f} MB")



end_time = time.time()
print(f'Training time per batches of size {batch_size}: {(end_time - start_time) / 10:.4f} seconds')

