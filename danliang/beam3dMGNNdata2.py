import numpy as np
import pickle
import beam3dM2


# =========================
# 构建 edge_index
# =========================
def build_edge_index(elems):
    edges = []
    for e in elems:
        i, j = int(e[0]), int(e[1])
        edges.append([i, j])
        edges.append([j, i])
    return np.array(edges).T


# =========================
# 构建 edge_attr
# =========================
def build_edge_attr(coords, elems, EA, EI):

    edge_attr = []

    for e in elems:
        i, j = int(e[0]), int(e[1])

        xi = coords[i]
        xj = coords[j]

        d = xj - xi
        L = np.linalg.norm(d)

        attr = [
            d[0], d[1], d[2],
            L,
            EA,
            EI,
            0.0
        ]

        edge_attr.append(attr)
        edge_attr.append(attr)

    return np.array(edge_attr)


# =========================
# Node features
# =========================
def build_node_features(coords, U, V, A, fixed_nodes):

    num_nodes = coords.shape[0]
    x = []

    for i in range(num_nodes):

        base = i * 6

        pos = coords[i]
        disp = U[base:base+3]
        rot  = U[base+3:base+6]
        vel  = V[base:base+6]
        acc  = A[base:base+6]

        is_fixed = 1.0 if i in fixed_nodes else 0.0

        feat = np.concatenate([
            pos, disp, rot, vel, acc, [is_fixed]
        ])

        x.append(feat)

    return np.array(x)


# =========================
# label
# =========================
def build_label(A):
    return A.reshape(-1, 6)


# =========================
# 主函数（🔥多仿真版本）
# =========================
def generate_dataset():

    dataset = []

    num_simulations = 20   # 🔥 关键：控制数据集规模
    stride = 5             # 🔥 时间下采样

    print("🚀 Generating dataset with multiple simulations...")

    for sim in range(num_simulations):

        print(f"\n=== Simulation {sim+1}/{num_simulations} ===")

        # =========================
        # 🔥 随机化物理参数（核心）
        # =========================

        # 随机弯矩
        beam3dM2.M0 = np.random.uniform(160000, 170000)

        # 随机阻尼
        beam3dM2.damping_scale = np.random.uniform(0.3, 0.35)

        # （可选）随机长度
        # beam3dM2.L_total = np.random.uniform(0.5, 1.5)

        # =========================
        # FEM 求解
        # =========================
        U_hist, coords, _, _, _ = beam3dM2.solve_dynamic()

        coords = np.array(coords)

        num_elements = coords.shape[0] - 1
        elems = np.array([[i, i+1] for i in range(num_elements)])

        # 材料参数
        E = 200e9
        r = 0.02
        A = np.pi * r**2
        I = np.pi * r**4 / 4

        EA = E * A
        EI = E * I

        edge_index = build_edge_index(elems)
        edge_attr  = build_edge_attr(coords, elems, EA, EI)

        fixed_nodes = [0]

        dt = 1e-4

        # =========================
        # 构建时间序列数据
        # =========================
        for i in range(2, len(U_hist), stride):

            U = U_hist[i]
            U_prev = U_hist[i-1]
            U_prev2 = U_hist[i-2]

            V = (U - U_prev) / dt
            A_vec = (U - 2*U_prev + U_prev2) / (dt**2)

            x = build_node_features(coords, U, V, A_vec, fixed_nodes)
            y = build_label(A_vec)

            data = {
                "x": x.astype(np.float32),
                "edge_index": edge_index.astype(np.int64),
                "edge_attr": edge_attr.astype(np.float32),
                "y": y.astype(np.float32)
            }

            dataset.append(data)

    print("\n✅ Total dataset size:", len(dataset))

    # =========================
    # 保存
    # =========================
    with open("beam_gnn_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    print("💾 Saved to beam_gnn_dataset.pkl")


# =========================
# run
# =========================
if __name__ == "__main__":
    generate_dataset()