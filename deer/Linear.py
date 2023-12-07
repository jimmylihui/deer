import equinox as eqx
import jax
import jax.numpy as jnp
import math
import jax.random as jrandom
class Linear(eqx.Module):
    W:list
    
    B:list
    alpha:float

    def __init__(self, layers,key,alpha=0.1):
        self.W=[]
        self.B=[]
        self.alpha=alpha
        wkey, bkey = jrandom.split(key, 2)
        for i in jnp.arange(0,len(layers)-1):
            lim = 1 / math.sqrt(layers[i])
            weight = jrandom.uniform(
                wkey, (layers[i], layers[i+1]), minval=-lim, maxval=lim
            )
            self.W.append(weight)

            bias=jrandom.uniform(
                bkey, (layers[i+1],), minval=-lim, maxval=lim
            )
            self.B.append(bias)
        
    
    def sigmoid(self, x):
        # sigmoid激活函数
        return 1.0 / (1 + jnp.exp(-x))

    def sigmoid_deriv(self, x):
        # sigmoid的导数
        return x * (1 - x)
    

    def __call__(self, X,y):

        # X=jnp.c_[X,jnp.ones(X.shape[0])]
        # for (x, target) in zip(X, y):
        self.fit_partial(X, y)



    def fit_partial(self,x,y):
        A=[jnp.atleast_2d(x)]
        for layer in jnp.arange(0,len(self.W)):
            net = A[layer].dot(self.W[layer])
            out=self.sigmoid(net)

            A.append(out)
        
        error = A[-1] - y

        D=[error*self.sigmoid_deriv(A[-1])]

        for layer in jnp.arange(len(A)-2,0,-1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        D=D[::-1]
        for layer in jnp.arange(0,len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])


# x_key, y_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)
# nn = NeuralNetwork([2, 2, 1],model_key)
# x = jax.random.normal(x_key, ( 2,))
# y = jax.random.normal(y_key, ( 1,))
# nn(x,y)



