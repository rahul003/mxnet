import mxnet as mx
import numpy as np

a = mx.nd.ones((3,2), dtype=np.float16)
print(a.handle)
print(a.astype('float32').handle)
print(a.astype('float16').handle)
