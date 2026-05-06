import numpy as np
import pickle
import beam3dM2


# ============================================================
# edge index
# ============================================================
def build_edge_index(elems):

    edges = []

    for e in elems:

        i, j = int(e[0]), int(e[1])

        edges.append([i, j])
        edges.append([j, i])

    return np.array(edges).T


# ============================================================
# edge attr
# ============================================================
def build_edge_attr(coords, elems, EA, EI):

    edge_attr = []

    for e in elems:

        i, j = int(e[0]), int(e[1])

        xi = coords[i]
        xj = coords[j]

        d = xj - xi

        L = np.linalg.norm(d)

        attr = [

            d[0],
            d[1],
            d[2],

            L,

            EA,
            EI,

            0.0

        ]

        edge_attr.append(attr)
        edge_attr.append(attr)

    return np.array(edge_attr)


# ============================================================
# velocity + acceleration
# ============================================================
def compute_velocity_acceleration(
        U_prev2,
        U_prev,
        U_curr,
        dt
):

    V = (U_curr - U_prev) / dt

    A = (
        U_curr
        - 2 * U_prev
        + U_prev2
    ) / (dt**2)

    return V, A


# ============================================================
# node features
# ============================================================
def build_node_features(

        coords,

        U,
        V,
        A,

        fixed_nodes,

        M0
):

    num_nodes = coords.shape[0]

    x = []

    for i in range(num_nodes):

        base = i * 6

        pos = coords[i]

        disp = U[base:base+3]

        rot = U[base+3:base+6]

        vel = V[base:base+6]

        acc = A[base:base+6]

        is_fixed = 1.0 if i in fixed_nodes else 0.0

        feat = np.concatenate([

            pos,       # 3

            disp,      # 3
            rot,       # 3

            vel,       # 6

            acc,       # 6

            [M0],      # 1

            [is_fixed] # 1

        ])

        x.append(feat)

    return np.array(x)


# ============================================================
# labels
# ============================================================
def build_labels(

        U_next,
        V_next,
        A_next
):

    num_nodes = len(U_next) // 6

    y = []

    for i in range(num_nodes):

        base = i * 6

        disp_next = U_next[base:base+3]

        rot_next = U_next[base+3:base+6]

        vel_next = V_next[base:base+6]

        acc_next = A_next[base:base+6]

        label = np.concatenate([

            disp_next,
            rot_next,

            vel_next,

            acc_next

        ])

        y.append(label)

    return np.array(y)


# ============================================================
# main
# ============================================================
def generate_dataset():

    # --------------------------------------------------------
    # dataset = trajectories
    # --------------------------------------------------------

    dataset = []

    num_simulations = 20

    stride = 1

    dt = 1e-4

    print("🚀 Generating trajectory dataset...")

    for sim in range(num_simulations):

        print(f"\n=== Simulation {sim+1}/{num_simulations} ===")

        # ====================================================
        # random parameters
        # ====================================================

        M0 = np.random.uniform(
            160000,
            170000
        )

        beam3dM2.M0 = M0

        beam3dM2.damping_scale = np.random.uniform(
            0.3,
            0.35
        )

        # ====================================================
        # FEM
        # ====================================================

        U_hist, coords, _, _, _ = beam3dM2.solve_dynamic()

        coords = np.array(coords)

        # ====================================================
        # graph
        # ====================================================

        num_elements = coords.shape[0] - 1

        elems = np.array([

            [i, i+1]

            for i in range(num_elements)

        ])

        # ====================================================
        # material
        # ====================================================

        E = 200e9

        r = 0.02

        A_area = np.pi * r**2

        I = np.pi * r**4 / 4

        EA = E * A_area

        EI = E * I

        edge_index = build_edge_index(elems)

        edge_attr = build_edge_attr(

            coords,
            elems,

            EA,
            EI

        )

        fixed_nodes = [0]

        # ====================================================
        # trajectory
        # ====================================================

        trajectory = []

        # ====================================================
        # time loop
        # ====================================================

        for t in range(2, len(U_hist)-1, stride):

            # ------------------------------------------------
            # current state
            # ------------------------------------------------

            U_prev2 = U_hist[t-2]

            U_prev = U_hist[t-1]

            U_curr = U_hist[t]

            V_curr, A_curr = compute_velocity_acceleration(

                U_prev2,
                U_prev,
                U_curr,
                dt

            )

            # ------------------------------------------------
            # next state
            # ------------------------------------------------

            U_next = U_hist[t+1]

            V_next, A_next = compute_velocity_acceleration(

                U_prev,
                U_curr,
                U_next,
                dt

            )

            # ------------------------------------------------
            # x_t
            # ------------------------------------------------

            x = build_node_features(

                coords,

                U_curr,
                V_curr,
                A_curr,

                fixed_nodes,

                M0

            )

            # ------------------------------------------------
            # y_t
            # ------------------------------------------------

            y = build_labels(

                U_next,
                V_next,
                A_next

            )

            # ------------------------------------------------
            # save timestep
            # ------------------------------------------------

            data = {

                "x": x.astype(np.float32),

                "y": y.astype(np.float32),

                "edge_index": edge_index.astype(np.int64),

                "edge_attr": edge_attr.astype(np.float32),

                "M0": np.float32(M0),

                "dt": np.float32(dt),

                # 🔥 VERY IMPORTANT
                "sim_id": sim,

                "time_id": t

            }

            trajectory.append(data)

        # ====================================================
        # save trajectory
        # ====================================================

        dataset.append(trajectory)

        print(
            f"Trajectory length = {len(trajectory)}"
        )

    # ========================================================
    # save
    # ========================================================

    print("\n✅ Total simulations:", len(dataset))

    total_steps = sum(len(tr) for tr in dataset)

    print("✅ Total timesteps:", total_steps)

    with open(

        "beam_dynamic_trajectory_dataset.pkl",

        "wb"

    ) as f:

        pickle.dump(dataset, f)

    print(
        "\n💾 Saved to beam_dynamic_trajectory_dataset.pkl"
    )


# ============================================================
# run
# ============================================================
if __name__ == "__main__":

    generate_dataset()