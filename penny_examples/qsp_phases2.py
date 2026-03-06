import jax
import jax.numpy as jnp
from jax import lax

import optax
import jaxopt

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

t = 8
do,de = 15,14
def target_func(x):
    return jnp.exp(-t*x)/2

I = jnp.eye(2, dtype=jnp.complex128)
H = jnp.array([
    [1,  1],
    [1, -1]
    ])/jnp.sqrt(2)
H_kron_I = jnp.kron(H,I)

def R(x):
    s = jnp.sqrt(1-x*x+0j)
    return jnp.array([
        [x,  s],
        [s, -x]
    ], dtype=jnp.complex128)

# def PI(phi):
#     return jnp.array([
#         [jnp.exp(1j*phi),  0],
#         [0, jnp.exp(-1j*phi)]
#     ], dtype=jnp.complex128)

def qsp_with_parity(Rx,phiset):
    init_U = jnp.eye(2, dtype=jnp.complex128)

    def body_fun(carry,phi):
        phases = jnp.array([jnp.exp(1j*phi), jnp.exp(-1j*phi)])
        U_rot = carry * phases
        return U_rot @ Rx, None

    last_U, _ = lax.scan(body_fun, init_U, phiset)

    return last_U

def qsp(x,phiset_o,phiset_e):
    Rx = R(x)
    U_o = qsp_with_parity(Rx,phiset_o)
    U_e = qsp_with_parity(Rx,phiset_e)

    C_U_o_mult_C_U_e = jnp.block([
        [U_o, jnp.zeros((2,2), dtype=jnp.complex128)],
        [jnp.zeros((2,2), dtype=jnp.complex128), U_e]
    ])

    U_qsp = H_kron_I @ C_U_o_mult_C_U_e @ H_kron_I

    return U_qsp
qsp_batched = jax.vmap(qsp, (0,None,None), 0)

BATCH_SIZE = 100
NUM_TRAIN_STEPS = 1000

xs = jnp.linspace(0,1,BATCH_SIZE)
ys = target_func(xs)

key = jax.random.PRNGKey(67)
k1, k2 = jax.random.split(key)

initial_params = {
    'phiset_o': jax.random.normal(k1, (do,)) * 0.01,
    'phiset_e': jax.random.normal(k2, (de,)) * 0.01,
}

def net(params, xs):
    phiset_o = params["phiset_o"]
    phiset_e = params["phiset_e"]
    result = qsp_batched(xs,phiset_o,phiset_e)
    return result[:,0,0].real

def loss(params, xs, ys):
    ys_pred = net(params, xs)
#     weights = 1.0 / (ys + 1e-4)
    loss_value = jnp.log(jnp.mean(jnp.square(ys_pred - ys)))
    return loss_value

def fit(params, optimizer):
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, xs, ys):
        loss_value, grads = jax.value_and_grad(loss)(params, xs, ys)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i in range(NUM_TRAIN_STEPS):
        params, opt_state, loss_value = step(params, opt_state, xs, ys)
        if i % 100 ==0:
            print(f"step {i}, loss: {loss_value}")

    return params, opt_state

# scheduler = optax.linear_schedule(0.1,0.001,NUM_TRAIN_STEPS)
optimizer = optax.adam(learning_rate=0.01)
params, _ = fit(initial_params, optimizer)

lbfgs = jaxopt.LBFGS(fun=loss, maxiter=1000, tol=1e-13, jit=True)
res = lbfgs.run(params, xs=xs, ys=ys)
params = res.params
print(f"L-BFGS loss: {res.state.value}")

optimizer = optax.adam(learning_rate=0.001)
params, _ = fit(params, optimizer)

lbfgs = jaxopt.LBFGS(fun=loss, maxiter=1000, tol=1e-13, jit=True)
res = lbfgs.run(params, xs=xs, ys=ys)
params = res.params
print(f"L-BFGS loss: {res.state.value}")

optimizer = optax.adam(learning_rate=0.0001)
params, _ = fit(params, optimizer)

lbfgs = jaxopt.LBFGS(fun=loss, maxiter=1000, tol=1e-13, jit=True)
res = lbfgs.run(params, xs=xs, ys=ys)
params = res.params
print(f"L-BFGS loss: {res.state.value}")

optimizer = optax.adam(learning_rate=0.00001)
params, _ = fit(params, optimizer)

lbfgs = jaxopt.LBFGS(fun=loss, maxiter=10000, tol=1e-15, jit=True)
res = lbfgs.run(params, xs=xs, ys=ys)
params = res.params
print(f"L-BFGS loss: {res.state.value}")

ys_pred = net(params, xs)
plt.plot(xs,ys,label="Target")
plt.plot(xs,ys_pred,label="QSP Fit", linestyle="--")
plt.legend()
plt.show()

# R = lambda x: np.array([
#     [x              , np.sqrt(1-x*x)],
#     [np.sqrt(1-x*x) , -x            ]
#     ], dtype='complex128')

# PI = lambda phi: np.array([
#     [np.exp(1j*phi),               0],
#     [0             , np.exp(-1j*phi)]
#     ], dtype='complex128')

# def qsp_with_parity(x,phiset):
#     Rx = R(x)
#     U_qsp = np.eye(2, dtype='complex128')
#     for phi in phiset:
#         U_qsp @= PI(phi)@Rx
#     return U_qsp
# 
# def qsp(x,phiset_o,phiset_e):
#     O = np.zeros((2,2), dtype='complex128')
#     I = np.eye(2, dtype='complex128')
#     U_qsp_o = qsp_with_parity(x,phiset_o)
#     U_qsp_e = qsp_with_parity(x,phiset_e)
# 
#     C_U_qsp_o = np.block([
#                 [U_qsp_o, O],
#                 [O      , I]
#                 ])
#     C_U_qsp_e = np.block([
#                 [I, O      ],
#                 [O, U_qsp_e]
#                 ])
# 
#     U_qsp = np.kron(H,I) @ C_U_qsp_o @ C_U_qsp_e @ np.kron(H,I)
#     return U_qsp
# 
# def re_qsp(x,phiset_o,phiset_e):
#     O = np.zeros((4,4), dtype='complex128')
#     I = np.eye(4, dtype='complex128')
#     U_qsp = qsp(x,phiset_o,phiset_e)
#     U_qsp_adj = np.conjugate(U_qsp).T
# 
#     C_U_qsp = np.block([
#               [U_qsp, O],
#               [O    , I]
#               ])
#     C_U_qsp_adj = np.block([
#                   [I, O        ],
#                   [O, U_qsp_adj]
#                   ])
# 
#     re_U_qsp = np.kron(H,I) @ C_U_qsp @ C_U_qsp_adj @ np.kron(H,I)
#     return re_U_qsp
