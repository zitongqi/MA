import numpy as np
import cupy as cp
from cupyx.scipy.sparse import coo_matrix
from cupyx.scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DTYPE = cp.float64

# =========================
# 参数
# =========================
L_total = 1.0
num_elements = 20
num_nodes = num_elements + 1
dof_per_node = 3
total_dofs = num_nodes * dof_per_node

r = 0.05
E = 200e9
rho = 7800

A = np.pi * r**2
EA = float(E * A)

# =========================
# 时间参数
# =========================
dt = 1e-5
T_total = 0.005
num_steps = int(T_total / dt)

beta = 0.25
gamma = 0.5

NEWTON_MAX_IT = 20
TOL = 1e-6

alpha = 1e-3  # 阻尼

# =========================
# 网格
# =========================
def build_mesh():
    xs = np.linspace(0, L_total, num_nodes)
    coords = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
    elems = np.array([[i, i+1] for i in range(num_elements)])
    return coords, elems

# =========================
# 质量矩阵
# =========================
def build_mass(nodes, elements):
    M = cp.zeros(total_dofs)
    for e in elements:
        n1, n2 = int(e[0]), int(e[1])
        x1, x2 = nodes[n1], nodes[n2]
        L = cp.linalg.norm(x2 - x1)
        m = rho * A * L / 2
        for n in [n1, n2]:
            base = n * dof_per_node
            M[base:base+3] += m
    return M

# =========================
# 单元刚度（轴向）
# =========================
def element_routine(x1, x2, u1, u2):
    L = cp.linalg.norm(x2 - x1)
    k = EA / L

    K = cp.zeros((6, 6))
    K[0,0] = k
    K[3,3] = k
    K[0,3] = -k
    K[3,0] = -k

    du = u2[0] - u1[0]
    f = k * du

    q = cp.zeros(6)
    q[0] = -f
    q[3] = f

    return K, q

# =========================
# 动力学求解（带历史）
# =========================
def solve_dynamic():

    coords_np, elems_np = build_mesh()
    nodes = cp.asarray(coords_np)
    elems = cp.asarray(elems_np)

    M = build_mass(nodes, elems)
    C = alpha * M

    U = cp.zeros(total_dofs)
    V = cp.zeros(total_dofs)
    A_vec = cp.zeros(total_dofs)

    fixed = list(range(dof_per_node))
    free = cp.array([i for i in range(total_dofs) if i not in fixed])

    load_dof = (num_nodes - 1) * dof_per_node + 0

    U_history = []

    for step in range(num_steps):

        t = step * dt

        F_ext = cp.zeros(total_dofs)
        F_ext[load_dof] = 10 * cp.sin(20 * t)

        # predictor
        U_pred = U + dt*V + dt**2*(0.5 - beta)*A_vec
        V_pred = V + dt*(1 - gamma)*A_vec

        U_iter = U_pred.copy()

        for _ in range(NEWTON_MAX_IT):

            rows, cols, data = [], [], []
            qi = cp.zeros(total_dofs)

            for e in elems:
                n1, n2 = int(e[0]), int(e[1])

                dofs = list(range(n1*dof_per_node, n1*dof_per_node+3)) + \
                       list(range(n2*dof_per_node, n2*dof_per_node+3))

                x1, x2 = nodes[n1], nodes[n2]
                u1 = U_iter[n1*dof_per_node:(n1*dof_per_node+3)]
                u2 = U_iter[n2*dof_per_node:(n2*dof_per_node+3)]

                K_e, q_e = element_routine(x1, x2, u1, u2)

                for i in range(6):
                    qi[dofs[i]] += q_e[i]
                    for j in range(6):
                        rows.append(int(dofs[i]))
                        cols.append(int(dofs[j]))
                        data.append(float(K_e[i, j]))

            rows = cp.asarray(rows, dtype=cp.int32)
            cols = cp.asarray(cols, dtype=cp.int32)
            data = cp.asarray(data, dtype=DTYPE)

            K = coo_matrix((data, (rows, cols)), shape=(total_dofs, total_dofs)).tocsr()

            M_term = M / (beta * dt**2)
            C_term = C / (beta * dt)

            K_eff = K + coo_matrix(cp.diag(M_term + C_term))

            R = F_ext - qi \
                - M * ((U_iter - U_pred)/(beta*dt**2)) \
                - C * ((U_iter - U_pred)/(beta*dt))

            R_free = R[free]
            K_free = K_eff[free][:, free]

            dU = spsolve(K_free, R_free)

            if cp.any(cp.isnan(dU)):
                print("NaN detected")
                break

            U_iter[free] += dU

            if cp.linalg.norm(R_free) < TOL:
                break

        A_new = (U_iter - U_pred) / (beta * dt**2)
        V_new = V_pred + gamma * dt * A_new

        U = U_iter
        V = V_new
        A_vec = A_new

        U_history.append(cp.asnumpy(U.copy()))

        if step % 50 == 0:
            print(f"step {step}/{num_steps}")

    return U_history, coords_np

# =========================
# 动画
# =========================
def animate_beam(U_history, coords):

    fig, ax = plt.subplots()

    x0 = coords[:, 0]
    y0 = coords[:, 1]

    ax.plot(x0, y0, 'b--', label="Original")

    line, = ax.plot([], [], 'r-o', lw=2)

    scale = 1e8

    ax.set_xlim(x0.min(), x0.max())
    ax.set_ylim(-0.1, 0.1)
    ax.set_title("Beam vibration")

    def update(frame):
        U = U_history[frame].reshape(-1, 3)
        deformed = coords + scale * U

        x = deformed[:, 0]
        y = deformed[:, 1]

        line.set_data(x, y)
        return line,

    ani = FuncAnimation(fig, update, frames=len(U_history), interval=30)

    plt.legend()
    plt.show()

# =========================
# 主程序
# =========================
if __name__ == "__main__":
    U_hist, coords = solve_dynamic()
    animate_beam(U_hist, coords)