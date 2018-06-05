#! python3

import numpy as np

def MAE(preds, labels):
    '''
    Mean Absolute Error 
    '''
    preds = preds.astype(int)
    diff = preds - labels
    return np.average( np.abs(diff) )

def MPAE(preds, labels):
    '''
    Mean Absolute Percentage Error
    '''
    preds = preds.astype(float)
    diff = np.divide( preds - labels, labels )
    return np.average( np.abs( diff ) )

if __name__ == '__main__':
    pass