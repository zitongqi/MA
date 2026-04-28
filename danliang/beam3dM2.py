import numpy as np
import cupy as cp
from cupyx.scipy.sparse import coo_matrix, eye as sparse_eye
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
dof_per_node = 6
total_dofs = num_nodes * dof_per_node

E = 200e9
nu = 0.3
rho = 7800

r = 0.02
A = np.pi * r**2
I = np.pi * r**4 / 4
J = 2.0 * I
G = E / (2*(1+nu))

EA, GA2, GA3 = E*A, G*A, G*A
GJ, EI2, EI3 = G*J, E*I, E*I

# =========================
# mesh
# =========================
def build_mesh():
    x = np.linspace(0, L_total, num_nodes)
    coords = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    elems = np.array([[i, i+1] for i in range(num_elements)])
    return coords, elems


# =========================
# full tangent 梁单元（完全保留）
# =========================

E1_VEC = cp.array([1.0, 0.0, 0.0], dtype=DTYPE)
I3_3   = cp.eye(3, dtype=DTYPE)

def skew_batch(a):
    z = cp.zeros((a.shape[0],), dtype=a.dtype)
    return cp.stack([
        cp.stack([z,       -a[:, 2],  a[:, 1]], axis=1),
        cp.stack([a[:, 2],  z,       -a[:, 0]], axis=1),
        cp.stack([-a[:, 1], a[:, 0],  z      ], axis=1)
    ], axis=1)

def normalize(v, eps=1e-30):
    n = cp.linalg.norm(v, axis=1, keepdims=True)
    n = cp.maximum(n, eps)
    return v / n

def initialLocalFrame_batch(x1, x2):
    r21 = x2 - x1
    L = cp.linalg.norm(r21, axis=1, keepdims=True)
    L = cp.maximum(L, 1e-30)
    ex = r21 / L

    Ne = x1.shape[0]
    global_z = cp.array([0.0, 0.0, 1.0], dtype=DTYPE)
    global_y = cp.array([0.0, 1.0, 0.0], dtype=DTYPE)

    dot_z = cp.abs(cp.sum(ex * global_z[None, :], axis=1))
    arbitrary = cp.where((dot_z < 0.9)[:, None],
                         cp.broadcast_to(global_z, (Ne, 3)),
                         cp.broadcast_to(global_y, (Ne, 3)))

    ey = normalize(cp.cross(arbitrary, ex))
    ez = cp.cross(ex, ey)
    return cp.stack([ex, ey, ez], axis=2)

def rodrigues_batch(T0, dtheta):
    theta = cp.linalg.norm(dtheta, axis=1)
    small = theta < 1e-14

    axis = cp.where(
        small[:, None],
        cp.array([1.0, 0.0, 0.0], dtype=DTYPE),
        dtheta / cp.maximum(theta[:, None], 1e-30)
    )

    K = skew_batch(axis)
    ct = cp.cos(theta)[:, None, None]
    st = cp.sin(theta)[:, None, None]

    I3 = I3_3[None,:,:]
    outer = axis[:, :, None] * axis[:, None, :]

    R = ct * I3 + (1.0 - ct) * outer + st * K
    R = cp.where(small[:, None, None], I3, R)

    return R @ T0

def beam_batch_Kt_qi(x1, x2, d1, d2, a1, a2,
                     EA, GA2, GA3, GJ, EI2, EI3):

    Ne = x1.shape[0]
    l0 = cp.linalg.norm(x2 - x1, axis=1)

    dth = 0.5 * (a1 + a2)
    T0  = initialLocalFrame_batch(x1, x2)
    T   = rodrigues_batch(T0, dth)
    TT  = cp.swapaxes(T, 1, 2)

    r21 = (x2 + d2) - (x1 + d1)

    eps_l  = (TT @ r21[:, :, None])[:, :, 0] / l0[:, None] - E1_VEC[None, :]
    dchi_l = (TT @ (a2 - a1)[:, :, None])[:, :, 0] / l0[:, None]

    N_loc = cp.stack([EA  * eps_l[:, 0],
                      GA2 * eps_l[:, 1],
                      GA3 * eps_l[:, 2]], axis=1)

    M_loc = cp.stack([GJ  * dchi_l[:, 0],
                      EI2 * dchi_l[:, 1],
                      EI3 * dchi_l[:, 2]], axis=1)

    n = (T @ N_loc[:, :, None])[:, :, 0]
    m = (T @ M_loc[:, :, None])[:, :, 0]

    S_r21 = skew_batch(r21)

    I3e = cp.broadcast_to(I3_3, (Ne, 3, 3))
    Z3e = cp.zeros((Ne, 3, 3), dtype=DTYPE)

    X_top = cp.concatenate([-I3e, 0.5*S_r21, I3e, 0.5*S_r21], axis=2)
    X_bot = cp.concatenate([ Z3e, -I3e,      Z3e, I3e], axis=2)
    X     = cp.concatenate([X_top, X_bot], axis=1)

    T_bar = cp.zeros((Ne,6,6), dtype=DTYPE)
    T_bar[:,0:3,0:3] = T
    T_bar[:,3:6,3:6] = T

    C_diag = cp.asarray([EA, GA2, GA3, GJ, EI2, EI3], dtype=DTYPE)
    C = cp.zeros((Ne,6,6), dtype=DTYPE)
    C[:, cp.arange(6), cp.arange(6)] = C_diag[None,:]

    tmp   = T_bar @ (C @ cp.swapaxes(T_bar,1,2))
    K_mat = (cp.swapaxes(X,1,2) @ (tmp @ X))
    K_mat = (K_mat.T / l0).T

    S_n = skew_batch(n)
    S_m = skew_batch(m)
    Y   = 0.5 * (S_r21 @ S_n)

    K_sigma1 = cp.zeros((Ne,12,12), dtype=DTYPE)
    K_sigma2 = cp.zeros((Ne,12,12), dtype=DTYPE)

    K_sigma1[:, 0:3,  3:6 ] =  0.5 * S_n
    K_sigma1[:, 0:3,  9:12] =  0.5 * S_n
    K_sigma1[:, 3:6,  3:6 ] =  0.5 * S_m
    K_sigma1[:, 3:6,  9:12] =  0.5 * S_m
    K_sigma1[:, 6:9,  3:6 ] = -0.5 * S_n
    K_sigma1[:, 6:9,  9:12] = -0.5 * S_n
    K_sigma1[:, 9:12, 3:6 ] = -0.5 * S_m
    K_sigma1[:, 9:12, 9:12] = -0.5 * S_m

    K_sigma2[:, 3:6,  0:3 ] = -0.5 * S_n
    K_sigma2[:, 3:6,  3:6 ] =  0.5 * Y
    K_sigma2[:, 3:6,  6:9 ] =  0.5 * S_n
    K_sigma2[:, 3:6,  9:12] =  0.5 * Y

    K_sigma2[:, 9:12, 0:3 ] = -0.5 * S_n
    K_sigma2[:, 9:12, 3:6 ] =  0.5 * Y
    K_sigma2[:, 9:12, 6:9 ] =  0.5 * S_n
    K_sigma2[:, 9:12, 9:12] =  0.5 * Y

    Kt_e = K_mat + K_sigma1 + K_sigma2

    qi_e = (cp.swapaxes(X,1,2) @
            cp.concatenate([n, m], axis=1)[:,:,None])[:,:,0]

    return Kt_e, qi_e


# =========================
# solver（动力学）
# =========================
def solve_dynamic():

    coords_np, elems_np = build_mesh()
    nodes = cp.asarray(coords_np)
    elems = cp.asarray(elems_np)

    base0 = (elems[:,0]*6)[:,None]+cp.arange(6)
    base1 = (elems[:,1]*6)[:,None]+cp.arange(6)
    dofs_e = cp.concatenate([base0,base1],1)

    rr = cp.repeat(dofs_e,12,1)
    cc = cp.tile(dofs_e,(1,12))
    rows,cols = rr.reshape(-1),cc.reshape(-1)

    fixed = list(range(6))
    free = cp.array([i for i in range(total_dofs) if i not in fixed])

    # 🔥 修改1：阻尼增强
    M = cp.ones(total_dofs)*1e-2
    C = 0.2*M

    U = cp.zeros(total_dofs)
    V = cp.zeros(total_dofs)
    A_vec = cp.zeros(total_dofs)

    dt=1e-4
    beta=0.25
    gamma=0.5

    U_hist=[]
    tip_hist=[]
    tip_v_hist=[]
    tip_a_hist=[]

    for step in range(1000):

        t = step*dt

        # 🔥 修改2：平滑弯矩加载（核心）
        F = cp.zeros(total_dofs)
        M0 = 160000.0
        t_ramp = 0.05

        if t < t_ramp:
            factor = t / t_ramp
        else:
            factor = 1.0

        F[(num_nodes-1)*6 + 4] = M0 * factor

        U_pred = U + dt*V + dt**2*(0.5-beta)*A_vec
        V_pred = V + dt*(1-gamma)*A_vec

        U_iter = U_pred.copy()

        for _ in range(20):

            x1 = nodes[elems[:,0]]
            x2 = nodes[elems[:,1]]

            d1 = U_iter[(elems[:,0]*6)[:,None]+cp.arange(3)]
            a1 = U_iter[(elems[:,0]*6)[:,None]+cp.arange(3,6)]
            d2 = U_iter[(elems[:,1]*6)[:,None]+cp.arange(3)]
            a2 = U_iter[(elems[:,1]*6)[:,None]+cp.arange(3,6)]

            Kt_e, qi_e = beam_batch_Kt_qi(
                x1,x2,d1,d2,a1,a2,
                EA,GA2,GA3,GJ,EI2,EI3
            )

            data = Kt_e.reshape(-1)
            K = coo_matrix((data,(rows,cols)),
                           shape=(total_dofs,total_dofs)).tocsr()

            qi = cp.zeros(total_dofs)
            cp.add.at(qi,dofs_e.reshape(-1),qi_e.reshape(-1))

            Mterm = M/(beta*dt**2)
            Cterm = C/(beta*dt)

            K_eff = K + coo_matrix(cp.diag(Mterm + Cterm)) \
                      + 1e-10*sparse_eye(total_dofs)

            R = F - qi \
                - M*((U_iter-U_pred)/(beta*dt**2)) \
                - C*((U_iter-U_pred)/(beta*dt))

            dU = spsolve(K_eff[free][:,free],R[free])
            dU = cp.clip(dU, -1e-3, 1e-3)

            U_iter[free]+=dU

            if cp.linalg.norm(dU) < 1e-6:
                break

        A_new = (U_iter-U_pred)/(beta*dt**2)
        V_new = V_pred + gamma*dt*A_new

        U,V,A_vec = U_iter,V_new,A_new

        U_hist.append(cp.asnumpy(U.copy()))
        tip_hist.append(float(U[(num_nodes-1)*6+2]))
        tip_v_hist.append(float(V[(num_nodes-1)*6+2]))
        tip_a_hist.append(float(A_vec[(num_nodes-1)*6+2]))

        if step % 100 == 0:
            print("step", step, "max disp =", cp.max(cp.abs(U)))

        # 🔥 修改3：自动收敛终止
        if step > 500 and cp.linalg.norm(V) < 1e-6:
            print("✅ 收敛到静态圆弧")
            break

    return U_hist, coords_np, tip_hist, tip_v_hist, tip_a_hist


# =========================
# 可视化
# =========================
def visualize(U_hist, coords, tip_hist, tip_v_hist, tip_a_hist):

    # 🔥 修改4：scale 修正（关键）
    scale = 1

    fig = plt.figure(figsize=(18,4))

    ax1 = fig.add_subplot(141, projection='3d')

    def update(i):
        ax1.clear()
        U = U_hist[i].reshape(-1,6)[:,:3]
        d = coords + scale*U

        ax1.plot(coords[:,0],coords[:,1],coords[:,2],'b--')
        ax1.plot(d[:,0],d[:,1],d[:,2],'r-')

        ax1.set_xlim(0,1)
        ax1.set_ylim(-0.2,0.2)
        ax1.set_zlim(-0.2,0.2)

    ani = FuncAnimation(fig,update,frames=len(U_hist),interval=30)

    ax2 = fig.add_subplot(142)
    ax2.plot(tip_hist)
    ax2.set_title("Displacement")

    ax3 = fig.add_subplot(143)
    ax3.plot(tip_v_hist)
    ax3.set_title("Velocity")

    ax4 = fig.add_subplot(144)
    ax4.plot(tip_a_hist)
    ax4.set_title("Acceleration")

    plt.tight_layout()
    plt.show()


# =========================
# run
# =========================
if __name__ == "__main__":
    U_hist, coords, tip_hist, tip_v_hist, tip_a_hist = solve_dynamic()
    visualize(U_hist, coords, tip_hist, tip_v_hist, tip_a_hist)