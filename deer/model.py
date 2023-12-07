import equinox as eqx
import jax
from functools import partial
from jax import jit
import optax

class NeuralNetwork(eqx.Module):
    layers: list
    extra_bias: jax.Array

    def __init__(self, input,hidden,output,key):
        key1, key2, key3 = jax.random.split(key, 3)
        # These contain trainable parameters.
        self.layers = [eqx.nn.Linear(input, hidden, key=key1),
                       eqx.nn.Linear(hidden, hidden, key=key2),
                       eqx.nn.Linear(hidden, output, key=key3)]
        # This is also a trainable parameter.
        self.extra_bias = jax.numpy.ones(output)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x) + self.extra_bias
    

class Linear(eqx.Module):
    nn:eqx.Module
    # optim:eqx.Module
   
    def __init__(self, input,hidden,output,key):
        self.nn=NeuralNetwork(input,hidden,output,key)
        # self.optim = optax.adam(learning_rate)
        
    def __call__(self, x,y):
        # params, static = eqx.partition(self.nn, eqx.is_array)
        # self.nn=eqx.combine(params, static)
        # opt_state=self.optim.init(self.nn)
        grads =_loss(self.nn, x, y)
        # updates, opt_state = self.optim.update(grads, opt_state)
        # self.nn = eqx.apply_updates(self.nn, updates)
        object.__setattr__(self, 'nn', _update(self.nn,grads))
        # self.nn=_update(self.nn,grads)
        return 0

    @jax.jit  # compile this function to make it run fast.
    @jax.grad  # differentiate all floating-point arrays in `model`.
    def loss(self,model, x, y):
        # pred_y = jax.vmap(model)(x)  # vectorise the model over a batch of data
        # return jax.numpy.mean((y - pred_y) ** 2)  # L2 loss\
        return _loss(model,x,y)
    
    def update(self,model,grads):
        # new_model = jax.tree_util.tree_map(lambda m, g: m - learning_rate * g, model, grads)

        return _update(model,grads)
    

def _update(model,grads):
    learning_rate=0.1
    new_model = jax.tree_util.tree_map(lambda m, g: m - learning_rate * g, model, grads)

    return new_model

@jax.jit
@jax.grad
def _loss(model, x, y):
    pred_y = jax.vmap(model)(x)  # vectorise the model over a batch of data
    return jax.numpy.mean((y - pred_y) ** 2)  # L2 loss
    



@jax.jit  # compile this function to make it run fast.
@jax.grad  # differentiate all floating-point arrays in `model`.
def loss(model, x, y):
    pred_y = jax.vmap(model)(x)  # vectorise the model over a batch of data
    return jax.numpy.mean((y - pred_y) ** 2)  # L2 loss

# x_key, y_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)
# # Example data
# x = jax.random.normal(x_key, (100, 2))
# y = jax.random.normal(y_key, (100, 2))
# model = NeuralNetwork(model_key)
# # Compute gradients
# grads = loss(model, x, y)
# # Perform gradient descent
# learning_rate = 0.1
# new_model = jax.tree_util.tree_map(lambda m, g: m - learning_rate * g, model, grads)


# model=model_1(model_key)
# model(x,y)




