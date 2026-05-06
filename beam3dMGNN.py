import os
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================
# device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# settings
# ============================================================
SEQ_LEN = 20
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-4

NODE_IN = 17     # pos(3) + u(6) + v(6) + M0(1) + fixed(1)
EDGE_IN = 7
OUT_DIM = 12     # u_next(6) + v_next(6)

HIDDEN = 256
LSTM_HIDDEN = 256


# ============================================================
# feature slicing
# ============================================================
def build_uv_input_features(x_full):
    """
    原始 x_full:
        0:3    pos
        3:9    u
        9:15   v
        15:21  a
        21     M0
        22     fixed

    新输入:
        pos, u, v, M0, fixed
    """

    pos = x_full[:, 0:3]
    u = x_full[:, 3:9]
    v = x_full[:, 9:15]
    M0 = x_full[:, 21:22]
    fixed = x_full[:, 22:23]

    return torch.cat([pos, u, v, M0, fixed], dim=1)


def build_uv_target_from_x(x_full):
    """
    target 直接从下一时间步的 x 里面取 u,v
    """

    return x_full[:, 3:15]


# ============================================================
# load trajectory dataset into tensors
# ============================================================
def load_uv_sequence_tensor_dataset(path, seq_len=20):

    with open(path, "rb") as f:
        raw_dataset = pickle.load(f)

    X_list = []
    Y_list = []
    sim_id_list = []
    target_id_list = []

    edge_index = None
    edge_attr = None

    for sim_id, trajectory in enumerate(raw_dataset):

        T = len(trajectory)

        if T <= seq_len + 1:
            continue

        for t in range(T - seq_len):

            seq_x = []

            for k in range(seq_len):

                d = trajectory[t + k]

                x_full = torch.tensor(
                    d["x"],
                    dtype=torch.float32
                )

                x = build_uv_input_features(x_full)

                seq_x.append(x)

                if edge_index is None:

                    edge_index = torch.tensor(
                        d["edge_index"],
                        dtype=torch.long
                    )

                    edge_attr = torch.tensor(
                        d["edge_attr"],
                        dtype=torch.float32
                    )

            target = trajectory[t + seq_len]

            x_target_full = torch.tensor(
                target["x"],
                dtype=torch.float32
            )

            y = build_uv_target_from_x(x_target_full)

            X_list.append(torch.stack(seq_x, dim=0))
            Y_list.append(y)

            sim_id_list.append(sim_id)
            target_id_list.append(t + seq_len)

    X = torch.stack(X_list, dim=0)      # [S, T, N, 17]
    Y = torch.stack(Y_list, dim=0)      # [S, N, 12]

    sim_ids = torch.tensor(sim_id_list, dtype=torch.long)
    target_ids = torch.tensor(target_id_list, dtype=torch.long)

    print("✅ X shape:", X.shape)
    print("✅ Y shape:", Y.shape)
    print("✅ Total sequences:", X.shape[0])
    print("✅ Num nodes:", X.shape[2])

    return X, Y, edge_index, edge_attr, sim_ids, target_ids


# ============================================================
# normalizer
# ============================================================
class TensorNormalizer:

    def __init__(self):

        self.x_mean = None
        self.x_std = None

        self.y_mean = None
        self.y_std = None

        self.edge_mean = None
        self.edge_std = None

    def fit(self, X_train, Y_train, edge_attr):

        all_x = X_train.reshape(-1, X_train.shape[-1])
        all_y = Y_train.reshape(-1, Y_train.shape[-1])

        self.x_mean = all_x.mean(dim=0)
        self.x_std = all_x.std(dim=0) + 1e-6

        self.y_mean = all_y.mean(dim=0)
        self.y_std = all_y.std(dim=0) + 1e-6

        self.edge_mean = edge_attr.mean(dim=0)
        self.edge_std = edge_attr.std(dim=0) + 1e-6

        # 坐标不归一化
        self.x_mean[0:3] = 0.0
        self.x_std[0:3] = 1.0

        # fixed flag 不归一化
        self.x_mean[-1] = 0.0
        self.x_std[-1] = 1.0

        print("✅ Normalizer fitted")

    def normalize_x(self, X):

        return (X - self.x_mean) / self.x_std

    def normalize_y(self, Y):

        return (Y - self.y_mean) / self.y_std

    def normalize_edge_attr(self, edge_attr):

        return (edge_attr - self.edge_mean) / self.edge_std

    def denorm_y(self, Y):

        return Y * self.y_std.to(Y.device) + self.y_mean.to(Y.device)

    def to(self, device):

        self.x_mean = self.x_mean.to(device)
        self.x_std = self.x_std.to(device)

        self.y_mean = self.y_mean.to(device)
        self.y_std = self.y_std.to(device)

        self.edge_mean = self.edge_mean.to(device)
        self.edge_std = self.edge_std.to(device)


# ============================================================
# torch dataset
# ============================================================
class BeamSequenceDataset(Dataset):

    def __init__(self, X, Y, sim_ids, target_ids, indices):

        self.X = X
        self.Y = Y
        self.sim_ids = sim_ids
        self.target_ids = target_ids
        self.indices = indices

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        return (
            self.X[real_idx],
            self.Y[real_idx],
            self.sim_ids[real_idx],
            self.target_ids[real_idx]
        )


# ============================================================
# split by simulation
# ============================================================
def split_by_simulation(sim_ids, train_ratio=0.8):

    unique_sims = sorted(sim_ids.unique().tolist())

    random.shuffle(unique_sims)

    split = int(len(unique_sims) * train_ratio)

    train_sims = set(unique_sims[:split])
    test_sims = set(unique_sims[split:])

    if len(test_sims) == 0:
        test_sims = set([unique_sims[-1]])
        train_sims = set(unique_sims[:-1])

    train_indices = []
    test_indices = []

    for i, s in enumerate(sim_ids.tolist()):

        if s in train_sims:
            train_indices.append(i)
        else:
            test_indices.append(i)

    print("Train simulations:", sorted(train_sims))
    print("Test simulations:", sorted(test_sims))
    print("Train sequences:", len(train_indices))
    print("Test sequences:", len(test_indices))

    return train_indices, test_indices


# ============================================================
# Batched GNN Layer
# ============================================================
class BatchedGNNLayer(nn.Module):

    def __init__(self, node_in, edge_in, hidden):

        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(node_in * 2 + edge_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(node_in + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

    def forward(self, x, edge_index, edge_attr):
        """
        x:
            [B, N, node_in]

        edge_index:
            [2, E]

        edge_attr:
            [E, edge_in]
        """

        B, N, _ = x.shape

        src = edge_index[0]
        dst = edge_index[1]

        x_i = x[:, dst, :]    # target node
        x_j = x[:, src, :]    # source node

        e = edge_attr.unsqueeze(0).expand(B, -1, -1)

        m_input = torch.cat([x_i, x_j, e], dim=-1)

        m = self.edge_mlp(m_input)    # [B, E, hidden]

        aggr = torch.zeros(
            B,
            N,
            m.shape[-1],
            device=x.device
        )

        aggr.index_add_(1, dst, m)

        h = torch.cat([x, aggr], dim=-1)

        return self.node_mlp(h)


# ============================================================
# GNN + LSTM model
# ============================================================
class FastDynamicsGNNLSTM(nn.Module):

    def __init__(
            self,
            node_in=17,
            edge_in=7,
            hidden=256,
            lstm_hidden=256,
            out_dim=12
    ):

        super().__init__()

        self.gnn = BatchedGNNLayer(
            node_in=node_in,
            edge_in=edge_in,
            hidden=hidden
        )

        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=lstm_hidden,
            batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(lstm_hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, X_seq, edge_index, edge_attr):
        """
        X_seq:
            [B, T, N, node_in]
        """

        B, T, N, F = X_seq.shape

        H_list = []

        for t in range(T):

            h_t = self.gnn(
                X_seq[:, t, :, :],
                edge_index,
                edge_attr
            )

            H_list.append(h_t)

        H = torch.stack(H_list, dim=1)   # [B, T, N, hidden]

        # LSTM 需要 [B*N, T, hidden]
        H = H.permute(0, 2, 1, 3).contiguous()
        H = H.view(B * N, T, -1)

        out, _ = self.lstm(H)

        last = out[:, -1, :]

        pred = self.decoder(last)

        pred = pred.view(B, N, -1)

        return pred


# ============================================================
# loss
# ============================================================
def compute_loss(pred, y):

    loss_u = torch.mean(
        (pred[:, :, 0:6] - y[:, :, 0:6]) ** 2
    )

    loss_v = torch.mean(
        (pred[:, :, 6:12] - y[:, :, 6:12]) ** 2
    )

    loss = 1.0 * loss_u + 0.1 * loss_v

    return loss, loss_u.detach(), loss_v.detach()


# ============================================================
# train epoch
# ============================================================
def train_epoch(model, loader, optimizer, edge_index, edge_attr):

    model.train()

    total_loss = 0.0
    total_u = 0.0
    total_v = 0.0

    for X_batch, Y_batch, _, _ in loader:

        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()

        pred = model(
            X_batch,
            edge_index,
            edge_attr
        )

        loss, loss_u, loss_v = compute_loss(pred, Y_batch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()
        total_u += loss_u.item()
        total_v += loss_v.item()

    n = len(loader)

    return total_loss / n, total_u / n, total_v / n


# ============================================================
# test epoch
# ============================================================
def test_epoch(model, loader, edge_index, edge_attr):

    model.eval()

    total_loss = 0.0
    total_u = 0.0
    total_v = 0.0

    with torch.no_grad():

        for X_batch, Y_batch, _, _ in loader:

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            pred = model(
                X_batch,
                edge_index,
                edge_attr
            )

            loss, loss_u, loss_v = compute_loss(pred, Y_batch)

            total_loss += loss.item()
            total_u += loss_u.item()
            total_v += loss_v.item()

    n = len(loader)

    return total_loss / n, total_u / n, total_v / n


# ============================================================
# visualize one-step time response
# ============================================================
def visualize_time_response(
        model,
        dataset,
        normalizer,
        edge_index,
        edge_attr,
        num_nodes_to_plot=3,
        response_steps=120,
        direction=2
):

    model.eval()

    sim_ids_all = []

    for i in range(len(dataset)):
        _, _, sim_id, _ = dataset[i]
        sim_ids_all.append(int(sim_id))

    sim_ids_unique = sorted(list(set(sim_ids_all)))

    chosen_sim = random.choice(sim_ids_unique)

    selected_indices = []

    for i in range(len(dataset)):

        _, _, sim_id, target_id = dataset[i]

        if int(sim_id) == chosen_sim:
            selected_indices.append(
                (i, int(target_id))
            )

    selected_indices = sorted(
        selected_indices,
        key=lambda x: x[1]
    )

    if len(selected_indices) > response_steps:

        start = random.randint(
            0,
            len(selected_indices) - response_steps
        )

        selected_indices = selected_indices[start:start + response_steps]

    else:

        response_steps = len(selected_indices)

    first_X, first_Y, _, _ = dataset[selected_indices[0][0]]

    num_nodes = first_Y.shape[0]

    selected_nodes = np.random.choice(
        num_nodes,
        num_nodes_to_plot,
        replace=False
    )

    print("Selected sim:", chosen_sim)
    print("Selected nodes:", selected_nodes)
    print("Direction:", ["x", "y", "z"][direction])

    gt_u = {n: [] for n in selected_nodes}
    pred_u = {n: [] for n in selected_nodes}

    gt_v = {n: [] for n in selected_nodes}
    pred_v = {n: [] for n in selected_nodes}

    with torch.no_grad():

        for idx, _ in selected_indices:

            X, Y, _, _ = dataset[idx]

            X = X.unsqueeze(0).to(device)
            Y = Y.unsqueeze(0).to(device)

            pred = model(
                X,
                edge_index,
                edge_attr
            )

            pred_real = normalizer.denorm_y(
                pred.squeeze(0)
            ).cpu()

            gt_real = normalizer.denorm_y(
                Y.squeeze(0)
            ).cpu()

            for n in selected_nodes:

                gt_u[n].append(
                    gt_real[n, direction].item()
                )

                pred_u[n].append(
                    pred_real[n, direction].item()
                )

                gt_v[n].append(
                    gt_real[n, 6 + direction].item()
                )

                pred_v[n].append(
                    pred_real[n, 6 + direction].item()
                )

    time = np.arange(response_steps)

    fig, axes = plt.subplots(
        num_nodes_to_plot,
        2,
        figsize=(14, 4 * num_nodes_to_plot)
    )

    if num_nodes_to_plot == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, n in enumerate(selected_nodes):

        axes[row, 0].plot(
            time,
            gt_u[n],
            label="GT"
        )

        axes[row, 0].plot(
            time,
            pred_u[n],
            "--",
            label="Pred"
        )

        axes[row, 0].set_title(
            f"Node {n} - u{['x','y','z'][direction]}(t)"
        )

        axes[row, 0].legend()

        axes[row, 1].plot(
            time,
            gt_v[n],
            label="GT"
        )

        axes[row, 1].plot(
            time,
            pred_v[n],
            "--",
            label="Pred"
        )

        axes[row, 1].set_title(
            f"Node {n} - v{['x','y','z'][direction]}(t)"
        )

        axes[row, 1].legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# main
# ============================================================
def main():

    file_path = os.path.join(
        os.path.dirname(__file__),
        "beam_dynamic_trajectory_dataset.pkl"
    )

    X, Y, edge_index, edge_attr, sim_ids, target_ids = load_uv_sequence_tensor_dataset(
        file_path,
        seq_len=SEQ_LEN
    )

    train_indices, test_indices = split_by_simulation(
        sim_ids,
        train_ratio=0.8
    )

    # ========================================================
    # normalizer
    # ========================================================
    normalizer = TensorNormalizer()

    normalizer.fit(
        X[train_indices],
        Y[train_indices],
        edge_attr
    )

    X = normalizer.normalize_x(X)
    Y = normalizer.normalize_y(Y)
    edge_attr = normalizer.normalize_edge_attr(edge_attr)

    normalizer.to(device)

    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    train_dataset = BeamSequenceDataset(
        X,
        Y,
        sim_ids,
        target_ids,
        train_indices
    )

    test_dataset = BeamSequenceDataset(
        X,
        Y,
        sim_ids,
        target_ids,
        test_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ========================================================
    # model
    # ========================================================
    model = FastDynamicsGNNLSTM(
        node_in=NODE_IN,
        edge_in=EDGE_IN,
        hidden=HIDDEN,
        lstm_hidden=LSTM_HIDDEN,
        out_dim=OUT_DIM
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR
    )

    train_losses = []
    test_losses = []

    train_disp_losses = []
    train_vel_losses = []

    test_disp_losses = []
    test_vel_losses = []

    # ========================================================
    # training
    # ========================================================
    for epoch in range(EPOCHS):

        train_loss, train_u, train_v = train_epoch(
            model,
            train_loader,
            optimizer,
            edge_index,
            edge_attr
        )

        test_loss, test_u, test_v = test_epoch(
            model,
            test_loader,
            edge_index,
            edge_attr
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        train_disp_losses.append(train_u)
        train_vel_losses.append(train_v)

        test_disp_losses.append(test_u)
        test_vel_losses.append(test_v)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(
            f"Train Total Loss: {train_loss:.6f} | "
            f"Disp Loss: {train_u:.6f} | "
            f"Vel Loss: {train_v:.6f}"
        )
        print(
            f"Test  Total Loss: {test_loss:.6f} | "
            f"Disp Loss: {test_u:.6f} | "
            f"Vel Loss: {test_v:.6f}"
        )
        print("-" * 60)

    # ========================================================
    # loss curve
    # ========================================================
    plt.figure(figsize=(9, 4))

    plt.plot(train_losses, label="Train Total")
    plt.plot(test_losses, label="Test Total")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4))

    plt.plot(train_disp_losses, label="Train Disp")
    plt.plot(test_disp_losses, label="Test Disp")

    plt.plot(train_vel_losses, label="Train Vel")
    plt.plot(test_vel_losses, label="Test Vel")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Component Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ========================================================
    # visualize
    # ========================================================
    visualize_time_response(
        model,
        test_dataset,
        normalizer,
        edge_index,
        edge_attr,
        num_nodes_to_plot=3,
        response_steps=120,
        direction=2
    )

    return model, normalizer


# ============================================================
# run
# ============================================================
if __name__ == "__main__":

    model, normalizer = main()