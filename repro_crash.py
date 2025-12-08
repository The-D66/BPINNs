import os
# Uncomment the next line to test the fix
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
print(f"TensorFlow Version: {tf.__version__}")

try:
    print("Creating a simple constant...")
    a = tf.constant([1.0, 2.0])
    print(f"Constant created: {a}")
    
    print("Performing a simple operation...")
    b = a * 2.0
    print(f"Result: {b}")
    
    print("Success!")
except Exception as e:
    print(f"Crash: {e}")
