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

# 🔥 为了能看到变形，降低刚度（可改回 200e9）
E = 2e7
nu = 0.3
rho = 7800

r = 0.05
A = np.pi * r**2
I = np.pi * r**4 / 4
J = 2.0 * I
G = E / (2*(1+nu))

EA, GA2, GA3 = E*A, G*A, G*A
GJ, EI2, EI3 = G*J, E*I, E*I

I3 = cp.eye(3)

# =========================
# mesh
# =========================
def build_mesh():
    x = np.linspace(0, L_total, num_nodes)
    coords = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    elems = np.array([[i, i+1] for i in range(num_elements)])
    return coords, elems

# =========================
# 工具函数
# =========================
def skew(v):
    z = cp.zeros((v.shape[0],))
    return cp.stack([
        cp.stack([z,-v[:,2],v[:,1]],1),
        cp.stack([v[:,2],z,-v[:,0]],1),
        cp.stack([-v[:,1],v[:,0],z],1)
    ],1)

def normalize(v):
    return v/(cp.linalg.norm(v,axis=1,keepdims=True)+1e-30)

def frame(x1,x2):
    r = x2-x1
    ex = normalize(r)
    z = cp.array([0,0,1])[None,:]
    ey = normalize(cp.cross(z,ex))
    ez = cp.cross(ex,ey)
    return cp.stack([ex,ey,ez],2)

# =========================
# beam
# =========================
def beam_batch(x1,x2,d1,d2):

    Ne = x1.shape[0]
    L = cp.linalg.norm(x2-x1,axis=1)

    T = frame(x1,x2)
    TT = cp.swapaxes(T,1,2)

    r = (x2+d2)-(x1+d1)
    eps = (TT @ r[:,:,None])[:,:,0]/L[:,None]

    N = EA*eps[:,0]
    n_local = cp.stack([N,0*N,0*N],axis=1)
    n = (T @ n_local[:,:,None])[:,:,0]

    S = skew(r)

    I3e = cp.broadcast_to(I3,(Ne,3,3))
    Z3e = cp.zeros((Ne,3,3))

    X_top = cp.concatenate([-I3e,0.5*S,I3e,0.5*S],2)
    X_bot = cp.concatenate([Z3e,-I3e,Z3e,I3e],2)
    X = cp.concatenate([X_top,X_bot],1)

    qi = (cp.swapaxes(X,1,2) @
          cp.concatenate([n,n],axis=1)[:,:,None])[:,:,0]

    # 梁刚度（含EI）
    K = cp.zeros((Ne,12,12))
    for i in range(Ne):
        l = L[i]
        k = cp.zeros((12,12))

        k[0,0]=k[6,6]=EA/l
        k[0,6]=k[6,0]=-EA/l

        k[2,2]=k[8,8]=12*EI2/l**3
        k[2,8]=k[8,2]=-12*EI2/l**3

        K[i]=k

    return K, qi

# =========================
# solver
# =========================
def solve():

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

    # mass
    M = cp.ones(total_dofs)*1e-3
    C = 1e-3*M

    U = cp.zeros(total_dofs)
    V = cp.zeros(total_dofs)
    A_vec = cp.zeros(total_dofs)

    dt=1e-4
    beta=0.25
    gamma=0.5

    U_hist=[]
    tip_hist=[]

    for step in range(300):

        t = step*dt

        F = cp.zeros(total_dofs)
        F[(num_nodes-1)*6+2] = -100*cp.sin(20*t)

        U_pred = U + dt*V + dt**2*(0.5-beta)*A_vec
        V_pred = V + dt*(1-gamma)*A_vec

        U_iter = U_pred.copy()

        for _ in range(20):

            x1 = nodes[elems[:,0]]
            x2 = nodes[elems[:,1]]

            d1 = U_iter[(elems[:,0]*6)[:,None]+cp.arange(3)]
            d2 = U_iter[(elems[:,1]*6)[:,None]+cp.arange(3)]

            Kt_e, qi_e = beam_batch(x1,x2,d1,d2)

            data = Kt_e.reshape(-1)
            K = coo_matrix((data,(rows,cols)),
                           shape=(total_dofs,total_dofs)).tocsr()

            qi = cp.zeros(total_dofs)
            cp.add.at(qi,dofs_e.reshape(-1),qi_e.reshape(-1))

            Mterm = M/(beta*dt**2)
            K_eff = K + coo_matrix(cp.diag(Mterm)) \
                      + 1e-8*sparse_eye(total_dofs)

            R = F - qi - M*((U_iter-U_pred)/(beta*dt**2))

            dU = spsolve(K_eff[free][:,free],R[free])
            U_iter[free]+=dU

            if cp.linalg.norm(R[free])<1e-5:
                break

        A_new = (U_iter-U_pred)/(beta*dt**2)
        V_new = V_pred + gamma*dt*A_new

        U,V,A_vec = U_iter,V_new,A_new

        U_hist.append(cp.asnumpy(U.copy()))
        tip_hist.append(float(U[-1]))

        if step % 50 == 0:
            print(f"step {step}, max disp = {cp.max(cp.abs(U))}")

    return U_hist, coords_np, tip_hist

# =========================
# 可视化
# =========================
def visualize(U_hist, coords, tip_hist):

    # 自动 scale
    max_disp = np.max(np.abs(U_hist))
    scale = 1.0 / (max_disp + 1e-12) * 0.2

    print("最大位移:", max_disp)
    print("scale:", scale)

    fig = plt.figure(figsize=(10,4))

    # ===== 3D 动画 =====
    ax1 = fig.add_subplot(121, projection='3d')

    def update(i):
        ax1.clear()
        U = U_hist[i].reshape(-1,6)[:,:3]
        d = coords + scale*U

        ax1.plot(coords[:,0],coords[:,1],coords[:,2],'b--')
        ax1.plot(d[:,0],d[:,1],d[:,2],'r-')

        ax1.set_xlim(0,1)
        ax1.set_ylim(-0.3,0.3)
        ax1.set_zlim(-0.3,0.3)

    ani = FuncAnimation(fig,update,frames=len(U_hist),interval=30)

    # ===== 位移曲线 =====
    ax2 = fig.add_subplot(122)
    ax2.plot(tip_hist)
    ax2.set_title("Tip displacement")
    ax2.set_xlabel("time step")
    ax2.set_ylabel("displacement")

    plt.show()

# =========================
# run
# =========================
if __name__ == "__main__":
    U_hist, coords, tip_hist = solve()
    visualize(U_hist, coords, tip_hist)