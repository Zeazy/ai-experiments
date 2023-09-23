from functools import partial

def quad(a, b, c, x):
    return a**2 + b*x + c

def make_quad(a, b, c):
    return partial(quad, a, b, c)

def mse(y_hat, y):
    return ((y_hat-y)**2).mean()

def quad_mse(a, b, c, x, y):
    f = make_quad(a, b, c)
    return mse(f(x), y)