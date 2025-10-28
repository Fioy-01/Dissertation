import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import pairwise_distances

# 随机采样
def random_sampling(pool_dataset, k):
    indices = np.random.choice(len(pool_dataset), k, replace=False)
    return indices.tolist()

# 不确定性采样（1 - 最大概率）
def uncertainty_sampling(model, pool_dataset, k):
    outputs = model.predict(pool_dataset)
    logits = torch.tensor(outputs.predictions)
    probs = torch.nn.functional.softmax(logits, dim=1)
    uncertainty = 1 - torch.max(probs, dim=1).values
    topk = torch.topk(uncertainty, k)
    return topk.indices.tolist()

# 熵采样
def entropy_sampling(model, pool_dataset, k):
    outputs = model.predict(pool_dataset)
    logits = torch.tensor(outputs.predictions)
    probs = torch.nn.functional.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    topk = torch.topk(entropy, k)
    return topk.indices.tolist()

# 核心集采样（K-Center Greedy）
def core_set_sampling(model, labeled_dataset, pool_dataset, k):
    # 提取表示向量
    rep_fn = lambda d: torch.tensor(model.predict(d).hidden_states[-1][:, 0, :])  # cls token
    labeled_vecs = rep_fn(labeled_dataset)
    pool_vecs = rep_fn(pool_dataset)

    # 计算 pairwise 距离
    dist_matrix = pairwise_distances(pool_vecs.numpy(), labeled_vecs.numpy())
    min_dist = dist_matrix.min(axis=1)
    selected = []

    for _ in range(k):
        idx = np.argmax(min_dist)
        selected.append(idx)
        dist_new = pairwise_distances(pool_vecs.numpy(), pool_vecs[idx:idx+1].numpy())
        min_dist = np.minimum(min_dist, dist_new[:, 0])

    return selected

