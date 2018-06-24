#! python3

import numpy as np

def chunks(x, n):
    return [x[i:i + n] for i in range(0, len(x), n)]

def shuffle_xy(x, y):
    assert(len(x) == len(y))
    a = x.copy()
    b = y.copy()  
    
    combined = list(zip(a, b))
    np.random.shuffle(combined)

    a[:], b[:] = zip(*combined)
    return a, b
