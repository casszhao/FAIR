import numpy as np
from sklearn.metrics import f1_score

num = 100
def func():
      global num
      num = 200
print(num)
func()
print(num)

