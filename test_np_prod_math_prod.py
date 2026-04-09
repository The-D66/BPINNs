import tensorflow as tf
import time
import numpy as np
import math

values = [tf.random.normal((100, 100)) for _ in range(50)]

# Just check what kind of object t.shape is
print(type(values[0].shape))
