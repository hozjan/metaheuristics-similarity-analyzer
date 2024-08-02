import numpy as np

def random_float_with_step(low, high, step, size=None, replace=True):
    steps = np.arange(low / step, high / step)
    random_steps = np.random.choice(steps, size=size, replace=replace)
    random_floats = random_steps * step
    return random_floats