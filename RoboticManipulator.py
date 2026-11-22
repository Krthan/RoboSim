import numpy as np
from typing import Literal
np.set_printoptions(precision=6, suppress=True)
from scipy.optimize import minimize

class Rotation:
    @staticmethod
    def RotationX(angle):
        rot = [[1, 0 , 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
        return np.array(rot)
    
    @staticmethod
    def RotationY(angle):
        rot = [[np.cos(angle), 0 , np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        return np.array(rot)
    
    @staticmethod
    def RotationZ(angle):
        rot = [[np.cos(angle), -np.sin(angle) , 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        return np.array(rot)
    

class Transform:
  def __init__(self, p: np.ndarray, R: np.ndarray):
    self.p = np.asarray(p, dtype=float).reshape(3, 1)
    self.R = np.asarray(R, dtype=float).reshape(3, 3)

  def getTransform(self) -> np.ndarray:
    top = np.hstack((self.R, self.p))
    bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
    return np.vstack((top, bottom))

  def __matmul__(self, other):
    if isinstance(other, Transform):
      T_new = self.getTransform() @ other.getTransform()
      return Transform(T_new[:3, 3:4], T_new[:3, :3])
    return NotImplemented

  def __repr__(self):
    with np.printoptions(precision=4, suppress=True):
      return str(self.getTransform())  

def getSkewSymmetric(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).ravel()
    assert p.size == 3
    S = np.zeros((3, 3), dtype=p.dtype)
    S[0, 1] = -p[2]
    S[1, 0] = p[2]
    S[0, 2] = p[1]
    S[2, 0] = -p[1]
    S[1, 2] = -p[0]
    S[2, 1] = p[0]
    
    return S
    
def getAdjointInv(T: Transform) -> np.ndarray:
    T = T.getTransform()
    R, p = T[:3, :3], T[:3, 3]
    p_skew = getSkewSymmetric(p)
    top = np.hstack((R.T, np.zeros((3, 3))))
    bottom = np.hstack((-R.T @ p_skew, R.T))
    
    return np.vstack((top, bottom))

def createAdjoint(V: np.ndarray) -> np.ndarray:
    """Computes 6x6 Lie bracket matrix [ad_V] for twist V = [w; v]."""
    V = np.asarray(V, dtype=float).ravel()
    w_skew = getSkewSymmetric(V[:3])
    v_skew = getSkewSymmetric(V[3:])
    ad_mat = np.zeros((6, 6))
    ad_mat[:3, :3] = w_skew
    ad_mat[3:, :3] = v_skew
    ad_mat[3:, 3:] = w_skew
    
    return ad_mat

def matrix_exp(S: np.ndarray, theta: float) -> Transform:
    """exp([S] * theta) for a screw axis S = [w; v], returned as a Transform."""
    S = np.asarray(S, dtype=float).ravel()
    assert S.size == 6, "S must have 6 elements"
    w = S[:3]
    v = S[3:]
    w_norm = np.linalg.norm(w)
    
    if w_norm < 1e-10:
        return Transform(v * theta, np.eye(3))
    
    w_hat = w / w_norm
    v_hat = v / w_norm
    phi = w_norm * theta

    W = getSkewSymmetric(w_hat)
    I = np.eye(3)

    # Rodrigues' formula
    R = I + np.sin(phi) * W + (1.0 - np.cos(phi)) * (W @ W)

    # G(phi) = I*phi + (1 - cos phi)[w] + (phi - sin phi)[w]^2
    G = I * phi + (1.0 - np.cos(phi)) * W + (phi - np.sin(phi)) * (W @ W)
    p = G @ v_hat

    return Transform(p, R)

def crba_algorithm(robot, q: np.ndarray) -> np.ndarray:
    """Composite Rigid Body Algorithm in body coordinates. Returns M(q), n x n."""
    links, joints = robot.links, robot.joints
    n = len(joints)
    q = np.asarray(q, dtype=float).ravel()
    Ad_inv = [getAdjointInv(joints[i].getrelativeTransform(links[i].M_rel_home, q[i])) for i in range(n)]
    
    Ic = [l.InertiaMatrix.copy() for l in links]
    
    for k in reversed(range(1, n)):
        Ic[k-1] += Ad_inv[k].T @ Ic[k] @ Ad_inv[k]
    
    M = np.zeros((n, n))
    
    for i in range(n):
        F = Ic[i] @ joints[i].S
        M[i, i] = (joints[i].S.T @ F).item()
        for j in range(i, 0, -1):
            F = Ad_inv[j].T @ F
            M[j-1, i] = M[i, j-1] = (joints[j-1].S.T @ F).item()
        
    return M
    

def aba_algorithm(robot, q, dq, tau, gravity):
    links, joints = robot.links, robot.joints
    n = len(joints)
    
    q = np.asarray(q, dtype=float).ravel()
    dq = np.asarray(dq, dtype=float).ravel()
    tau = np.asarray(tau, dtype=float).ravel()
    gravity = np.asarray(gravity, dtype=float).ravel()
    
    Ad_inv = [getAdjointInv(joints[i].getrelativeTransform(links[i].M_rel_home, q[i])) for i in range(n)]
    S = [j.S.ravel() for j in joints]
    
    v = [None] * n
    c = [None] * n
    IA = [links[i].InertiaMatrix.copy() for i in range(n)]
    pA = [None] * n
    
    v_prev = np.zeros(6)
    for i in range(n):
        v[i] = Ad_inv[i] @ v_prev + S[i] * dq[i]
        c[i] = createAdjoint(v[i]) @ (S[i] * dq[i])
        pA[i] = -createAdjoint(v[i]).T @ (links[i].InertiaMatrix @ v[i])
        v_prev = v[i]
    
    U = [None] * n
    D = np.zeros(n)
    u = np.zeros(n)
    for i in reversed(range(n)):
        U[i] = IA[i] @ S[i]
        D[i] = S[i] @ U[i]
        if D[i] < 1e-12:
            raise ValueError(
                f"Joint '{joints[i].name}' has zero effective inertia (D={D[i]:.3g}): "
                "the link and everything beyond it are massless, so its acceleration is undefined."
            )
        u[i] = tau[i] - S[i] @ pA[i]
        if i > 0:
            Ia = IA[i] - np.outer(U[i], U[i]) / D[i]
            pa = pA[i] + Ia @ c[i] + U[i] * (u[i] / D[i])
            IA[i-1] += Ad_inv[i].T @ Ia @ Ad_inv[i]
            pA[i-1] += Ad_inv[i].T @ pa
            
    
    a_prev = np.concatenate([np.zeros(3), -gravity])
    ddq = np.zeros(n)
    for i in range(n):
        a_p = Ad_inv[i] @ a_prev + c[i]
        ddq[i] = (u[i] - U[i] @ a_p) / D[i]
        a_prev = a_p + S[i] * ddq[i]
    
    return ddq
        
    
class Link:
    def __init__(self, name: str, mass: float, inertia: np.ndarray, M_rel_home: Transform):
        self.name = name
        self.mass = mass
        self.inertia = np.asarray(inertia, dtype=float).reshape(3, 3)
        self.M_rel_home = M_rel_home
        
        self.InertiaMatrix = np.zeros((6, 6))
        self.InertiaMatrix[:3, :3] = self.inertia
        self.InertiaMatrix[3:, 3:] = self.mass * np.eye(3)
    
    def getInertiaMatrix(self) -> np.ndarray:
        return self.InertiaMatrix
              

class Joint:
    def __init__(self, name: str, screw_axis: np.ndarray, parent: str = None, child: str = None, limits: dict = None):
        """
        screw_axis -> np array (6, 1) xi = [w_x, w_y, w_z, v_x, v_y, v_z]
        
        """
        self.name = name
        self.S = np.asarray(screw_axis, dtype=float).reshape(6, 1)
        self.parent = parent
        self.child = child
        self.limits = limits
    
    def getrelativeTransform(self, M_rel_home: Transform, q:float) -> Transform:
        return M_rel_home @ matrix_exp(self.S, q)


class Robot:
    def __init__(self, name: str, joints: list[Joint], links: list[Link]):
        if len(joints) != len(links):
            raise ValueError("Number of joints must match the number of links")
    
        self.name = name
        self.joints = joints
        self.links = links
    
    def body_jacobian(self, q, k=None):
        """
        Returns body jacobian of body
        
        """
        q = np.asarray(q, dtype=float).ravel()
        n = len(self.joints)
        k = n-1 if k is None else k
        J = np.zeros((6, n))
        for i in range(k+1):
            T_i = self.joints[i].getrelativeTransform(self.links[i].M_rel_home, q[i])
            J = getAdjointInv(T_i) @ J
            J[:, i] = self.joints[i].S.ravel()
        
        return J
    
    def space_jacobian(self, q, k=None):
        """
        Returns the space jacobian of the robot 
        
        """
        
        J_body = self.body_jacobian(q)
        p, R = self.forward_kinematics(q)
        J_s = np.linalg.inv(getAdjointInv(Transform(p, R))) @ J_body
        
        return J_s
    
    def forward_kinematics(self, q_values: list[float]) -> tuple[(np.ndarray, np.ndarray)]:
        """
        Returns the position and rotation of the end-effector with respect to the base frame of the robot given the joint angles array
        
        """
        n = len(self.joints)
        q = np.asarray(q_values, dtype=float).ravel()
        T = Transform(np.zeros(3), np.eye(3))
        for i in range(n):
            T = T @ self.joints[i].getrelativeTransform(self.links[i].M_rel_home, q[i])
        return T.p, T.R
    
    def inverse_kinematics(self, pos: list[float], initial_guess: np.ndarray, method: str = "lm", tol: float = 1e-6, Nmax: int = 500, **kwargs) -> np.ndarray:
        """
        Returns the joint angles of the robot when given the spatial position of the end-effector
        
        """
        pos = np.asarray(pos, dtype=float).ravel()
        q = np.asarray(initial_guess, dtype=float).ravel().copy()
        
        method = method.lower()
        
        if method == "lm":
            return self._ik_lm(pos, q, tol=tol, Nmax=Nmax, **kwargs)
        
        elif method == "opt":
            return self._ik_opt(pos, q, tol=tol, Nmax=max(Nmax, 1000), **kwargs)
        
        else:
            raise ValueError(f"{method} is not a valid method, choose between 'lm' or 'opt")
        
        
    def _ik_lm(self, pos: np.ndarray, q0: np.ndarray, tol: float, Nmax: int, step_size: float = 0.5, LM_factor: float = 1e-2) -> np.ndarray:
        """
        Levenberg-Marquardt (damped least squares) solver

        """
        q = q0.copy()
        for _ in range(Nmax):
            x_curr, R = self.forward_kinematics(q)
            err = pos - x_curr.ravel()
            if np.linalg.norm(err) < tol:
                return q
            
            Jp = R @ self.body_jacobian(q)[3:, :]
            dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + LM_factor**2 * np.eye(3), err)
            q = q + step_size * dq
        
        raise RuntimeError(f"IK did not converge in {Nmax} iterations (residual {np.linalg.norm(err):.3g})")
    
    def _ik_opt(self, pos: np.ndarray, q0: np.ndarray, tol: float, Nmax: int, solver: str = "SLSQP", bounds: list[tuple[float, float]] | None = None, reg_weight: float = 1e-8):
    
        """Nonlinear optimization solver using analytical gradients."""
        
        def cost_and_grad(q: np.ndarray) -> tuple[float, np.ndarray]:
            x_curr, R = self.forward_kinematics(q)
            err = x_curr.ravel() - pos
            
            cost = 0.5 * np.dot(err, err) + 0.5 * reg_weight * np.dot(q - q0, q - q0)
            
            Jp = R @ self.body_jacobian(q)[3:, :]
            grad = Jp.T @ err + reg_weight * (q - q0)
            
            return cost, grad
        
        res = minimize(
            fun=cost_and_grad,
            x0=q0,
            method=solver,
            jac=True,  
            bounds=bounds,
            options={"maxiter": Nmax, "ftol": (1e-2 * tol) ** 2},
        )

        x_final, _ = self.forward_kinematics(res.x)
        final_err = np.linalg.norm(pos - x_final.ravel())
        if final_err > tol:
            raise RuntimeError(
                f"Optimization IK did not converge (residual {final_err:.3g}, msg: {res.message})"
            )

        return res.x
        
    
    def forward_dynamics(self, algorithm: Literal["crba", "aba"], q: list[float], dq: list[float], tau: list[float], gravity: np.ndarray = np.array([0.0, 0.0, -9.81])):
        n = len(self.joints)
        q = np.asarray(q, dtype=float).ravel()
        dq = np.asarray(dq, dtype=float).ravel()
        tau = np.asarray(tau, dtype=float).ravel()

        if algorithm == "crba":
            # bias = C(q, dq)dq + g(q)
            bias = self.inverse_dynamics(q, dq, np.zeros(n), gravity).ravel()
            M = crba_algorithm(self, q)
            for i in range(n):
                if M[i, i] < 1e-12:
                    raise ValueError(
                        f"Joint '{self.joints[i].name}' has zero effective inertia (M[{i},{i}]={M[i, i]:.3g}): "
                        "the link and everything beyond it are massless, so its acceleration is undefined."
                    )
            try:
                return np.linalg.solve(M, tau - bias)
            except np.linalg.LinAlgError as e:
                raise ValueError("Mass matrix is singular; forward dynamics is undefined for this configuration.") from e
        elif algorithm == "aba":
            return aba_algorithm(self, q, dq, tau, gravity)
        raise ValueError(f"{algorithm} is not a valid algorithm")
    
    def inverse_dynamics(self, q: list[float], dq: list[float], ddq: list[float], gravity: np.ndarray = np.array([0.0, 0.0, -9.81])):
        n = len(self.joints)
        q = np.asarray(q, dtype=float).ravel()
        dq = np.asarray(dq, dtype=float).ravel()
        ddq = np.asarray(ddq, dtype=float).ravel()
        g_vec = np.asarray(gravity, dtype=float).reshape(3, 1)
        
        v = [np.zeros((6, 1)) for _ in range(n+1)]
        a = [np.zeros((6, 1)) for _ in range(n+1)]
        f_body = [np.zeros((6, 1)) for _ in range(n+1)]
        
        a[0][3:, :] = -g_vec
        
        Ad_T_inv_list = []
        
        for i in range(n):
            S_i = self.joints[i].S #motion subspace of joint i
            T_prev_curr = self.joints[i].getrelativeTransform(self.links[i].M_rel_home, q[i])
            
            Ad_inv = getAdjointInv(T_prev_curr)
            Ad_T_inv_list.append(Ad_inv)
            
            v[i+1] = Ad_inv @ v[i] + S_i * dq[i]
            a[i+1] = Ad_inv @ a[i] + createAdjoint(v[i+1]) @ (S_i * dq[i]) + S_i * ddq[i]
            f_body[i+1] = self.links[i].InertiaMatrix @ a[i+1] - createAdjoint(v[i+1]).T @ (self.links[i].InertiaMatrix @ v[i+1])
        
        tau = np.zeros((n, 1))
        f_i = [np.zeros((6, 1)) for _ in range(n)]
        
        for i in reversed(range(n)):
            S_i = self.joints[i].S
            if i < n - 1:
                f_i[i] = f_body[i + 1] + Ad_T_inv_list[i + 1].T @ f_i[i + 1]
            else:
                f_i[i] = f_body[i + 1]
                
            tau[i, 0] = (S_i.T @ f_i[i]).item()
        
        return tau     
    
    def getGravityTerm(self, q: list[float], gravity: np.ndarray = None) -> np.ndarray:
        """Returns g(q)."""
        zeros = np.zeros(len(self.joints))
        gravity = np.array([0.0, 0.0, -9.81]) if gravity is None else gravity
        return self.inverse_dynamics(q, zeros, zeros, gravity=gravity)
    
    def getCoriolisTerm(self, q: list[float], dq: list[float]) -> np.ndarray:
        """Returns C(q, dq)dq"""
        zeros = np.zeros(len(self.joints))
        return self.inverse_dynamics(q, dq, zeros, gravity=np.zeros(3))
    
    def getMassMatrixTerm(self, q: list[float]) -> np.ndarray:
        """Returns n x n Inertia Matrix M(q)."""
        n = len(self.joints)
        M = np.zeros((n, n))
        zeros = np.zeros(n)
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            M[:, i] = self.inverse_dynamics(q, zeros, e_i, gravity=np.zeros(3)).ravel()
        return M
    
    






    