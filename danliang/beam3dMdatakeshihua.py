# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# # =========================
# # 加载数据
# # =========================
# def load_dataset(path="beam_gnn_dataset.pkl"):
#     with open(path, "rb") as f:
#         dataset = pickle.load(f)
#     print("✅ Loaded dataset, size =", len(dataset))
#     return dataset


# # =========================
# # 可视化单个样本（3D梁）
# # =========================
# def visualize_sample(data, scale=1.0):

#     x = data["x"]          # [21,22]
#     edge_index = data["edge_index"]
#     y = data["y"]          # acceleration [21,6]

#     coords = x[:, 0:3]
#     disp   = x[:, 3:6]

#     deformed = coords + scale * disp

#     # 加速度大小（用于颜色）
#     acc = y[:, 0:3]
#     acc_mag = np.linalg.norm(acc, axis=1)

#     fig = plt.figure(figsize=(8,6))
#     ax = fig.add_subplot(111, projection='3d')

#     # 原始梁
#     ax.plot(coords[:,0], coords[:,1], coords[:,2],
#             'b--', label="Original")

#     # 变形梁（颜色表示加速度）
#     p = ax.scatter(
#         deformed[:,0],
#         deformed[:,1],
#         deformed[:,2],
#         c=acc_mag,
#         cmap='jet',
#         s=40
#     )

#     ax.plot(deformed[:,0], deformed[:,1], deformed[:,2],
#             'r-', label="Deformed")

#     fig.colorbar(p, ax=ax, label="|Acceleration|")

#     ax.set_title("Beam Deformation + Acceleration")
#     ax.legend()

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")

#     plt.show()


# # =========================
# # 查看特征分布
# # =========================
# def plot_feature_distribution(dataset):

#     all_x = np.concatenate([d["x"] for d in dataset], axis=0)

#     plt.figure(figsize=(10,4))
#     plt.hist(all_x[:,3], bins=50)  # ux
#     plt.title("Displacement distribution (ux)")
#     plt.show()


# # =========================
# # 查看加速度分布
# # =========================
# def plot_acc_distribution(dataset):

#     all_y = np.concatenate([d["y"] for d in dataset], axis=0)

#     acc_mag = np.linalg.norm(all_y[:,0:3], axis=1)

#     plt.figure(figsize=(10,4))
#     plt.hist(acc_mag, bins=50)
#     plt.title("Acceleration magnitude distribution")
#     plt.show()


# # =========================
# # 主函数
# # =========================
# def main():

#     dataset = load_dataset()

#     # ====== 看一个样本 ======
#     idx = np.random.randint(len(dataset))
#     print("Visualizing sample:", idx)

#     visualize_sample(dataset[idx], scale=1)

#     # ====== 数据统计 ======
#     plot_feature_distribution(dataset)
#     plot_acc_distribution(dataset)


# # =========================
# # run
# # =========================
# if __name__ == "__main__":
#     main()


import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# load
# ============================================================
def load_dataset(

        path="beam_dynamic_trajectory_dataset.pkl"
):

    with open(path, "rb") as f:

        dataset = pickle.load(f)

    print("✅ Loaded trajectories:", len(dataset))

    total_steps = sum(len(tr) for tr in dataset)

    print("✅ Total timesteps:", total_steps)

    return dataset


# ============================================================
# visualize one timestep
# ============================================================
def visualize_timestep(

        data,

        scale=1.0
):

    x = data["x"]

    y = data["y"]

    coords = x[:,0:3]

    disp = x[:,3:6]

    acc_next = y[:,12:15]

    deformed = coords + scale * disp

    acc_mag = np.linalg.norm(
        acc_next,
        axis=1
    )

    # ========================================================
    # plot
    # ========================================================

    fig = plt.figure(figsize=(10,7))

    ax = fig.add_subplot(
        111,
        projection='3d'
    )

    # original
    ax.plot(

        coords[:,0],
        coords[:,1],
        coords[:,2],

        'b--',

        label='Original'

    )

    # deformed
    ax.plot(

        deformed[:,0],
        deformed[:,1],
        deformed[:,2],

        'r-',

        linewidth=2,

        label='Deformed'

    )

    # acceleration
    p = ax.scatter(

        deformed[:,0],
        deformed[:,1],
        deformed[:,2],

        c=acc_mag,

        cmap='jet',

        s=50

    )

    fig.colorbar(

        p,

        ax=ax,

        label='|Acceleration_next|'

    )

    M0 = data["M0"]

    sim_id = data["sim_id"]

    time_id = data["time_id"]

    ax.set_title(

        f"Simulation={sim_id}  "
        f"Time={time_id}  "
        f"M0={M0:.2f}"

    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()

    plt.show()


# ============================================================
# visualize trajectory
# ============================================================
def visualize_trajectory(

        trajectory,

        scale=1.0,

        num_frames=10
):

    total_steps = len(trajectory)

    step_ids = np.linspace(

        0,
        total_steps-1,
        num_frames

    ).astype(int)

    fig = plt.figure(figsize=(12,8))

    ax = fig.add_subplot(
        111,
        projection='3d'
    )

    for i, step in enumerate(step_ids):

        data = trajectory[step]

        x = data["x"]

        coords = x[:,0:3]

        disp = x[:,3:6]

        deformed = coords + scale * disp

        deformed = deformed - deformed[0]

        ax.plot(

            deformed[:,0],
            deformed[:,1],
            deformed[:,2],

            label=f't={step}'

        )

    ax.set_title("Trajectory Evolution")

    ax.legend()

    plt.show()


# ============================================================
# displacement distribution
# ============================================================
def plot_displacement_distribution(dataset):

    all_x = []

    for trajectory in dataset:

        for d in trajectory:

            all_x.append(d["x"])

    all_x = np.concatenate(all_x, axis=0)

    ux = all_x[:,3]

    plt.figure(figsize=(8,4))

    plt.hist(ux, bins=50)

    plt.title("Displacement Distribution")

    plt.xlabel("ux")

    plt.ylabel("Count")

    plt.show()


# ============================================================
# acceleration distribution
# ============================================================
def plot_acc_distribution(dataset):

    all_y = []

    for trajectory in dataset:

        for d in trajectory:

            all_y.append(d["y"])

    all_y = np.concatenate(all_y, axis=0)

    acc = all_y[:,12:15]

    acc_mag = np.linalg.norm(
        acc,
        axis=1
    )

    plt.figure(figsize=(8,4))

    plt.hist(acc_mag, bins=50)

    plt.title("Acceleration Distribution")

    plt.xlabel("|a|")

    plt.ylabel("Count")

    plt.show()


# ============================================================
# M0 distribution
# ============================================================
def plot_M0_distribution(dataset):

    M0_list = []

    for trajectory in dataset:

        M0 = trajectory[0]["M0"]

        M0_list.append(M0)

    plt.figure(figsize=(8,4))

    plt.hist(M0_list, bins=20)

    plt.title("M0 Distribution")

    plt.xlabel("M0")

    plt.ylabel("Count")

    plt.show()


# ============================================================
# trajectory response
# ============================================================
def plot_node_response(

        trajectory,

        node_id=20
):

    disp_hist = []

    vel_hist = []

    acc_hist = []

    for d in trajectory:

        x = d["x"]

        # displacement
        u = x[node_id,3]

        # velocity
        v = x[node_id,9]

        # acceleration
        a = x[node_id,15]

        disp_hist.append(u)

        vel_hist.append(v)

        acc_hist.append(a)

    time = np.arange(len(trajectory))

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10,8)
    )

    axes[0].plot(time, disp_hist)

    axes[0].set_title(
        f'Node {node_id} Displacement'
    )

    axes[1].plot(time, vel_hist)

    axes[1].set_title(
        f'Node {node_id} Velocity'
    )

    axes[2].plot(time, acc_hist)

    axes[2].set_title(
        f'Node {node_id} Acceleration'
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# main
# ============================================================
def main():

    dataset = load_dataset()

    # ========================================================
    # choose trajectory
    # ========================================================

    sim_id = np.random.randint(len(dataset))

    trajectory = dataset[sim_id]

    print(f"\n🎯 Selected simulation: {sim_id}")

    print("Trajectory length:", len(trajectory))

    # ========================================================
    # random timestep
    # ========================================================

    t = np.random.randint(len(trajectory))

    print("Random timestep:", t)

    visualize_timestep(

        trajectory[t],

        scale=1.0

    )

    # ========================================================
    # visualize trajectory evolution
    # ========================================================

    visualize_trajectory(

        trajectory,

        scale=5.0,

        num_frames=15

    )

    # ========================================================
    # node response
    # ========================================================

    plot_node_response(

        trajectory,

        node_id=20

    )

    # ========================================================
    # distributions
    # ========================================================

    plot_displacement_distribution(dataset)

    plot_acc_distribution(dataset)

    plot_M0_distribution(dataset)


# ============================================================
# run
# ============================================================
if __name__ == "__main__":

    main()