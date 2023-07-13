import jax
import jax.test_util
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp as solve_ivp_scipy
from deer.seq1d import solve_ivp, matmul_recursive


jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

def test_matmul_recursive():
    nsamples = 100
    ny = 4

    # generate random matrix with shape (nsamples - 1, ny, ny)
    key = jax.random.PRNGKey(0)
    subkey1, subkey2, subkey3, key = jax.random.split(key, 4)
    mats = jax.random.normal(subkey1, (nsamples - 1, ny, ny), dtype=jnp.float64) / 3
    vecs = jax.random.normal(subkey2, (nsamples - 1, ny), dtype=jnp.float64)
    y0 = jax.random.normal(subkey3, (ny,), dtype=jnp.float64)

    # generate the result using for loop
    result = jnp.zeros((nsamples, ny), dtype=jnp.float64)
    result = result.at[0].set(y0)
    for i in range(nsamples - 1):
        result = result.at[i + 1].set(mats[i] @ result[i] + vecs[i])
    # generate the result using matmul_recursive
    result2 = matmul_recursive(mats, vecs, y0)
    assert jnp.allclose(result, result2)

def test_solve_ivp():
    ny = 4
    dtype = jnp.float64
    key = jax.random.PRNGKey(0)
    subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    A0 = (jax.random.uniform(subkey1, shape=(ny, ny), dtype=dtype) * 2 - 1) / ny ** 0.5
    A1 = jax.random.uniform(subkey2, shape=(ny, ny), dtype=dtype) / ny ** 0.5
    npts = 10000  # TODO: investigate why npts=1000 make nans
    tpts = jnp.linspace(0, 1.0, npts, dtype=dtype)  # (ntpts,)
    y0 = jax.random.uniform(subkey3, shape=(ny,), dtype=dtype)

    A0_np = np.array(A0)
    A1_np = np.array(A1)
    tpts_np = np.array(tpts)
    y0_np = np.array(y0)

    def func(y: jnp.ndarray, x: jnp.ndarray, params) -> jnp.ndarray:
        # x: (1,) is time
        # y: (ny,)
        # returns: (ny,)
        A0, A1 = params
        yy = jnp.tanh(A1 @ jax.nn.relu(A0 @ y))
        dfdy = -6 * yy - 10 * y ** 3 + 30 * jnp.sin(6 * x) + 3
        return dfdy

    def func_np(t: np.ndarray, y: np.ndarray, A0_np: np.ndarray, A1_np: np.ndarray) -> np.ndarray:
        yy = np.tanh(A1_np @ np.maximum(A0_np @ y, 0))
        dfdy = -6 * yy - 10 * y ** 3 + 30 * np.sin(6 * t) + 3
        return dfdy

    params = (A0, A1)
    params_np = (A0_np, A1_np)
    yt = solve_ivp(func, y0, tpts[..., None], params, tpts)  # (ntpts, ny)
    yt_np = solve_ivp_scipy(func_np, (tpts_np[0], tpts_np[-1]), y0_np, t_eval=tpts_np, args=params_np, rtol=1e-10, atol=1e-10).y.T

    # import matplotlib.pyplot as plt
    # plt.plot(tpts, yt[..., 0])
    # plt.plot(tpts, yt_np[..., 0])
    # # plt.plot(tpts, (yt - yt_np)[..., 0])
    # plt.savefig("test.png")
    # plt.close()

    assert jnp.allclose(yt, yt_np, atol=1e-6)

    # check the gradients
    def get_loss(y0, params):
        yt = solve_ivp(func, y0, tpts[..., None], params, tpts)  # (ntpts, ny)
        return jnp.sum(yt ** 2, axis=0)  # only sum over time
    jax.test_util.check_grads(
        get_loss, (y0, params), order=1, modes=['rev'],
        # atol, rtol, eps following torch.autograd.gradcheck
        atol=1e-5, rtol=1e-3, eps=1e-6)