# only use to test random code

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# together 3 data A = prob B = pred

data_num = 3
label_num = 9

for i in range(label_num):
      print(i)
      if i == 0:
            label_name = 'label_' + str(i) + '_'
            array_A = np.array([label_name + 'A', label_name + 'A', label_name + 'A'])
            array_B = np.array([label_name + 'B', label_name + 'B', label_name + 'B'])
      else:
            label_name = 'label_' + str(i) + '_'
            temp_A = np.array([label_name + 'A', label_name + 'A', label_name + 'A'])
            temp_B = np.array([label_name + 'B', label_name + 'B', label_name + 'B'])

            array_A = np.concatenate([array_A, temp_A])
            array_B = np.concatenate([array_B, temp_B])

print(array_A)
print(array_B)

print('------------------------')


A = np.transpose(array_A.reshape(label_num, data_num)).tolist()
B = np.transpose(array_B.reshape(label_num, data_num)).tolist()

df = pd.DataFrame(list(zip(A,B)),
                               columns =['A', 'B'])
print(df)



