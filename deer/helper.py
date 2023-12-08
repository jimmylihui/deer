
import math
import functools
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax  # https://github.com/deepmind/optax
from jax.numpy import einsum
import equinox as eqx


@eqx.filter_value_and_grad
def compute_loss(model, x, y):
    pred_y = jax.vmap(model)(x)
    # Trains with respect to binary cross-entropy
    return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))

optim = optax.adam(0.01)
@eqx.filter_jit
def make_step(model, x, y, opt_state):
    loss, grads = compute_loss(model, x, y)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

# @eqx.filter_jit
# @functools.partial(jax.jit, static_argnames=("structure"))
def Linear(x,y,parameter):
    # dim_1=structure[0]
    # dim_2=structure[1]
    parameter=jnp.reshape(parameter,(2,2))
    alpha=0.1
    x=jnp.atleast_2d(x)
    parameter=jnp.atleast_2d(parameter)
    y=jnp.atleast_2d(y)
    #x[batch, input_dimension], y[batch, output dimension], parameter[batch,input, output]
    forward=einsum('bi,io->bo',x,parameter)
    forward=sigmoid(forward)
    error=y-forward

    D=error*sigmoid_deriv(forward)
    # delta=D.dot(parameter.T)
    
    # jax.debug.print("parameter :{parameter}", parameter=parameter[0])
    # parameter+= -alpha*x.T.dot(D)
    parameter+= -alpha*einsum('bi,bo->io',x,D)
    # jax.debug.print("parameter after:{parameter}", parameter=parameter[0])
    parameter=jnp.reshape(parameter,(4))
    return parameter

def Linear_2(x,y,parameter):
    alpha=100
    x=jnp.atleast_2d(x)
    #x[batch, input_dimension], y[batch, output dimension], parameter[batch,input, output]
    forward=einsum('bi,io->bo',x,parameter)
    forward=sigmoid(forward)
    error=y-forward

    D=error*sigmoid_deriv(forward)
    # delta=D.dot(parameter.T)
    a=x.T.dot(D)
    # jax.debug.print("parameter :{parameter}", parameter=parameter[0])
    parameter+= -alpha*x.T.dot(D)
    # jax.debug.print("parameter after:{parameter}", parameter=parameter[0])
    return parameter,error


@functools.partial(jax.jit, static_argnames=( "model"))
def loop(parameter,input,output,model):
    error_list=[]
    
    for i in range(input.shape[0]):
        x=input[i:i+1,:]
        y=output[i:i+1,:]
        parameter=model(x,y,parameter)
        # error_list.append(error)
        # jax.debug.print("error:{error}", error=error[0])
        
    return parameter

    


@eqx.filter_jit
def sigmoid(x):
        
    return 1.0 / (1 + jnp.exp(-x))

@eqx.filter_jit
def sigmoid_deriv(x):
        
    return x * (1 - x)

# x_key, y_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)
# x = jax.random.normal(x_key, ( 10,2))
# y = jax.random.normal(y_key, ( 10,1))
# parameter=jax.random.normal(model_key, ( 10,2,1))
# # for i in range(10):
# #     parameter,error=loop(parameter,x,y,Linear)
# parameter,error=Linear(x,y,parameter)


    