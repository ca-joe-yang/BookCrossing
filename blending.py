#! python3

import os
import numpy as np

fname_list = os.listdir('result')
fname_list = [ fname for fname in fname_list if fname.endswith('.csv') and fname != 'blending.csv' ]

result = []
for fname in fname_list:
	x = np.genfromtxt(os.path.join('result', fname))
	result.append(x)

result = np.array(result)
result = np.average(result, axis=0)
np.savetxt('result/blending.csv', result.astype(int), fmt='%d')