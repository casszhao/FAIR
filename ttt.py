import numpy as np
from sklearn.metrics import f1_score

li = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
array = np.array((li))
tr = [0,0,1,0]
label_index = 0
x = array[:, label_index]

print(x)



f1_score(tr, x)

