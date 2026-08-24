import os
import cvxpy as cp
import numpy as np
import uaibot as ub
from scipy.linalg import expm
from aux_functions import log_SE3
from setup import create_scenario, plot_dist_min, plot_u_xi

LOG_BARRIER_SCALE = 1e4


class LogBarrierControlProblem:
    def __init__(self, num_constraints, mu):
        self.mu = mu
        self.u = cp.Variable((6, 1), name="u")
        self.A = cp.Parameter((num_constraints, 6), name="A")
        self.b = cp.Parameter((num_constraints, 1), name="b")
        self.u_d = cp.Parameter((6, 1), name="u_d")

        self.g = self.A @ self.u - self.b
        objective = cp.sum_squares(self.u - self.u_d) - mu * cp.sum(cp.log(LOG_BARRIER_SCALE * self.g))
        self.problem = cp.Problem(cp.Minimize(objective))

        self.solve_count = 0
        self.success_count = 0
        self.status_list = []
        self.min_slack_list = []

    def solve(self, A, b, u_d):
        A_value = np.asarray(A, dtype=float)
        b_value = np.asarray(b, dtype=float).reshape((-1, 1))
        u_d_value = np.asarray(u_d, dtype=float).reshape((6, 1))

        self.A.value = A_value
        self.b.value = b_value
        self.u_d.value = u_d_value

        if self.u.value is None:
            self.u.value = u_d_value

        self.solve_count += 1
        self.problem.solve(solver=cp.CLARABEL, warm_start=True)

        status = self.problem.status
        self.status_list.append(status)
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or self.u.value is None:
            return None, status, None

        u_value = np.matrix(self.u.value)
        slack = A_value @ np.asarray(u_value) - b_value
        min_slack = float(np.min(slack))
        self.min_slack_list.append(min_slack)

        if min_slack <= 0:
            return None, f"{status}_domain_violation", min_slack

        self.success_count += 1
        return u_value, status, min_slack

def cmpt_lambda_terms(objA,objB,H,xi,h,eps):
    
    #Extract:
    sA = H[0:3,-1]
    vA = xi[0:3,-1]
    wA = xi[3:6,-1]
    
    #Put objA in the pose, temporarily
    htm_A = np.matrix(objA.htm)
    objA.set_ani_frame(H)
    
    #Compute dist
    a_star, b_star, lambda_AB, _ = objA.compute_dist(obj = objB, h = h, eps = eps, tol=1e-6, no_iter_max=6000)
    
    #Compute D_xi_lambda_AB
    D_xi_lambda_AB = np.matrix(np.zeros((1,6)))
    D_xi_lambda_AB[:,0:3] = (a_star-b_star).T/(1e-6+lambda_AB)
    D_xi_lambda_AB[:,3:6] = (ub.Utils.S(b_star-sA)*(a_star-b_star)).T/(1e-6+lambda_AB)
     
    #Compute (d/dt) D_xi_lambda_AB
    psi_A_a = vA + ub.Utils.S(wA)*(a_star-sA)
    psi_A_b = vA + ub.Utils.S(wA)*(b_star-sA)
    J_Pi_A = np.matrix(objA.cpp_obj.projection_jacobian(b_star,h,eps))
    J_Pi_B = np.matrix(objB.cpp_obj.projection_jacobian(a_star,h,eps))
    d_Pi_A_dt = psi_A_a - J_Pi_A*psi_A_b

        
    a_star_dot = np.linalg.inv(np.identity(3)-J_Pi_A*J_Pi_B)*d_Pi_A_dt
    b_star_dot = J_Pi_B*a_star_dot
    
    d_D_xi_Lambda_AB_dt = np.matrix(np.zeros((1,6)))
    d_D_xi_Lambda_AB_dt[:,0:3] = (a_star_dot-b_star_dot).T
    d_D_xi_Lambda_AB_dt[:,3:6] = (ub.Utils.S(b_star-sA)*(a_star_dot-b_star_dot)+ub.Utils.S(b_star_dot-vA)*(a_star-b_star)).T
    
    d_D_xi_lambda_AB_dt = (d_D_xi_Lambda_AB_dt - (D_xi_lambda_AB*xi)*D_xi_lambda_AB)/(1e-6+lambda_AB)

    #Put back the initial pose of the object
    objA.set_ani_frame(htm_A)
    
    return lambda_AB, D_xi_lambda_AB, d_D_xi_lambda_AB_dt
    
def cmpt_control(H,xi,obj_robot,list_obs,u_d,h=0.05,eps=0.01,eta=0.3,lambda_min=0.01,xi_lim=0.08,optimizer=None):
    falhou = False
    A = np.matrix(np.zeros((0,6)))
    b = np.matrix(np.zeros((0,1)))
    
    #Add the constraints for obstacle avoidance
    for obs in list_obs:
        lambda_RO, D_xi_lambda_RO, d_D_xi_lambda_RO_dt = cmpt_lambda_terms(obj_robot,obs,H,xi,h,eps)
        
        ff = -d_D_xi_lambda_RO_dt*xi
        d_lambda_RO_dt = D_xi_lambda_RO*xi
        b_temp = ff - 2*eta*d_lambda_RO_dt-eta*eta*(lambda_RO-lambda_min)
        
        A = np.vstack([A,D_xi_lambda_RO])
        b = np.vstack([b,b_temp])
        
        
    #Add limits for xi
    A = np.vstack([A,np.identity(6)])
    b = np.vstack([b,-xi_lim*np.ones((6,1))])
    A = np.vstack([A,-np.identity(6)])
    b = np.vstack([b,-xi_lim*np.ones((6,1))]) 
    
    #Solve the log-barrier problem
    try:
        if optimizer is None:
            raise ValueError("optimizer must be a LogBarrierControlProblem")
        u, status, min_slack = optimizer.solve(A, b, u_d)
        if u is None:
            raise RuntimeError(f"CLARABEL status: {status}, min_slack: {min_slack}")
    except Exception as exc:
        falhou = True
        print("\n Log barrier falhou!  ")
        print("Motivo: ", exc)
        sim.save(address=os.path.dirname(__file__),
                file_name="se3_teste"
                )
        return 0, falhou


    return u, falhou
        
def cmpt_control_reactive(H, xi, H_d, kc=0.5):
#Spatial acceleration to reach a constant target pose Hd    
    
    def ext(M):
        return np.matrix(np.diag(M)).T
    
    s = H[0:3,-1]
    s_d = H_d[0:3,-1]
    Q = H[0:3,0:3]
    Q_d = H_d[0:3,0:3]
    
    w = xi[3:6,-1]
    Sx = ub.Utils.S([1,0,0])
    Sy = ub.Utils.S([0,1,0])
    Sz = ub.Utils.S([0,0,1])
    Sw = ub.Utils.S(w)
    
    r = np.matrix(np.zeros((6,1)))
    r[0:3,-1] = s-s_d
    r[3:6,-1] = ext(np.identity(3)-Q_d.T*Q)
    
    # #(d/dt) r = D_xi_r*xi
    D_xi_r = np.matrix(np.zeros((6,6)))
    D_xi_r[0:3,:] = np.hstack([np.identity(3), np.zeros((3,3))])
    ax = ext(-Q_d.T*Sx*Q)
    ay = ext(-Q_d.T*Sy*Q)
    az = ext(-Q_d.T*Sz*Q)
    D_xi_r[3:6,:] = np.hstack([np.zeros((3,3)), ax, ay, az]) 
    
    #d_D_xi_r_dt = (d/dt) D_xi_r
    d_D_xi_r_dt = np.matrix(np.zeros((6,6)))
    dax = ext(-Q_d.T*Sx*Sw*Q)
    day = ext(-Q_d.T*Sy*Sw*Q)
    daz = ext(-Q_d.T*Sz*Sw*Q)
    d_D_xi_r_dt[3:6,:] = np.hstack([np.zeros((3,3)), dax, day, daz]) 
    
    #Compute u_d
    u_d = ub.Utils.dp_inv(D_xi_r)*(-d_D_xi_r_dt*xi-2*kc*D_xi_r*xi-kc*kc*r) 

    return u_d, r


# =========================

def eval_xid_from_state(state_htm, htm_path, xi, kt1, kt2, kt3, kn1, kn2, dt):
    
    xid, tangent, normal, dist, idx = ub.Robot.vector_field_SE3(
        state=state_htm,
        curve=htm_path,
        kt1=kt1,
        kt2=kt2,
        kt3=kt3,
        kn1=kn1,
        kn2=kn2,
        ds=dt,
        delta=1e-2,
    )

    xid = np.asarray(xid, dtype=float).reshape(6, 1)
    xid[0:3, :] = xid[0:3, :] + ub.Utils.S(xid[3:6, :]) @ state_htm[0:3, -1].reshape(3, 1)
        
    return xid , dist, idx


def propagate_htm(htm, xi, dt_step):
    p = htm[0:3, 3].reshape(3, 1)
    R = htm[0:3, 0:3]

    v = np.asarray(xi[0:3]).reshape(3, 1)
    w = np.asarray(xi[3:6]).reshape(3, 1)

    R_next = expm(ub.Utils.S(w) * dt_step) @ R
    p_next = p + v * dt_step

    htm_next = np.eye(4)
    htm_next[0:3, 0:3] = R_next
    htm_next[0:3, 3] = p_next.flatten()

    return np.matrix(htm_next)

def compute_ud( curr_state, htm_path, xi, kt1, kt2, kt3, kn1, kn2, Kv):
    
        # Reference twist
        xid, dist, idx = eval_xid_from_state(
            state_htm=curr_state,
            htm_path=htm_path,
            xi=xi,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            dt=dt_num,
        )

        # Numerical approximation of reference twist derivative 
        htm_plus = propagate_htm(curr_state, xid, dt_num)
        htm_minus = propagate_htm(curr_state, xid, -dt_num)

        xid_plus, _, _ = eval_xid_from_state(
            state_htm=htm_plus,
            htm_path=htm_path,
            xi =xi,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            dt=dt_num,
        )

        xid_minus, _, _ = eval_xid_from_state(
            state_htm=htm_minus,
            htm_path=htm_path,
            xi = xi,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            dt=dt_num,
        )

        xid_dot = (xid_plus - xid_minus) / (2.0 * dt_num)


    

        return  ((xid_dot - Kv * (xi - xid)), dist, idx)

##############################
#     Scenario Setup         #
##############################

scenario = create_scenario()
sim = scenario.sim
robot_body = scenario.robot_body
robot_body_copy = scenario.robot_body_copy
robot_UAV = scenario.robot_UAV
all_obs = scenario.all_obs
htm_path = scenario.htm_path
htm_target = scenario.htm_target
ball_tr = scenario.ball_tr
H = scenario.H
xi = scenario.xi
dt = scenario.dt
dt_num = scenario.dt_num
t_max = scenario.t_max
simular_movimento = scenario.simular_movimento

##############################
#     Control Parameters     #
##############################


kt1 = 12.3
kt2 = .3      
kt3 = 1
       
kn1 = .10
kn2 = .04


Kv = 20

param_eta =  1.2
param_obs_delta = 0.01
mu = 5e-4

# generalized distance parameters
use_generalized_distance = True

if use_generalized_distance:
    dist_param_h   = 0.05
    dist_param_eps = 0.03
else:
    dist_param_h   = 0
    dist_param_eps = 0

num_control_constraints = len(all_obs) + 12
control_optimizer = LogBarrierControlProblem(num_control_constraints, mu)

##############################
#     Simulation Settings    #
##############################



min_ec_dist = []
xi_list = []
u_list = []
time_list = []
error = []
last_err = 1
idx = 0

##############################
#      Simulation Loop       #
##############################

falhou = False
final = True
path_followed = []

if simular_movimento:
    for k in range(int(t_max / dt)):
        if last_err < 0.025:
            print("last d(H) : ", last_err)
            break
        
        t = k * dt
        
        if idx > 0.72 * len(htm_path):
            if final:
                print("Reativo: ",t)
                final = False
            ud, _ = cmpt_control_reactive(H, xi, htm_target, kc=0.5)
        else:
            ud, dist, idx = compute_ud(H, htm_path, xi, kt1, kt2, kt3, kn1, kn2, Kv) 


        u, falhou = cmpt_control(
            H,
            xi,
            robot_body_copy,
            all_obs,
            ud,
            dist_param_h,
            dist_param_eps,
            param_eta,
            param_obs_delta,
            xi_lim=1,
            optimizer=control_optimizer,
        )
        if falhou:
            break
        
        #############################
        #       Apply control       #
        #############################
        xi = xi + u*dt
        H = propagate_htm(H, xi, dt)
        
        robot_UAV.add_ani_frame(time = t, htm = H)
        ball_tr.add_ani_frame(time = t, htm=htm_path[idx])
        
        # some useful data
        ec_dist = []
        for obs in all_obs:
            _, _, dist, _ = robot_body.compute_dist(obs)
            ec_dist.append(dist)
            
        min_ec_dist.append(min(ec_dist))
        u_list.append(u)     
        xi_list.append(xi)
        time_list.append(t)
        path_followed.append(H)
        error.append(np.linalg.norm(log_SE3(ub.Utils.inv_htm(H) @ htm_target)))
        last_err = error[-1]
        
##############################
#          Results           #
##############################
print("Log barrier mu: ", mu)
print("CLARABEL successful solves: ", f"{control_optimizer.success_count}/{control_optimizer.solve_count}")
if error:
    print("Final d(H): ", last_err)
    print("Converged to target tolerance: ", last_err < 0.025)
if control_optimizer.min_slack_list:
    print("Minimum log-barrier domain slack: ", min(control_optimizer.min_slack_list))
status_counts = {
    status: control_optimizer.status_list.count(status)
    for status in set(control_optimizer.status_list)
}
print("CLARABEL status counts: ", status_counts)

sim.save(
    address=os.path.dirname(__file__),
    file_name="se3_teste",
)


plot_dist_min(
    min_ec_dist,
    time_list,
    file_name="minimum_distance.pdf",
    labels=None,
    xlabel="Time (s)",
    ylabel=r"$d_{min}$",
    show_plot=False,
    title=None
)


max_index = (int) (27.5/dt)
plot_u_xi(
    u_list[: max_index],
    xi_list[:max_index],
    time_list[:max_index],
    file_name="u_xi_combined.pdf",
    labels_u=[
        r'$u_{v_x}$', r'$u_{v_y}$', r'$u_{v_z}$',
        r'$u_{\omega_x}$', r'$u_{\omega_y}$', r'$u_{\omega_z}$'
    ],
    labels_xi=[
        r'$v_x$', r'$v_y$', r'$v_z$',
        r'$\omega_x$', r'$\omega_y$', r'$\omega_z$'
    ]
)
