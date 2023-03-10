# importing the tensorflow package
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
print(tf.test.is_built_with_cuda())
# print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
import sys

import tensorflow.keras
# import pandas as pd
# import sklearn as sk
import tensorflow as tf

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
# print()
# print(f"Python {sys.version}")
# print(f"Pandas {pd.__version__}")
# print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

from tensorflow.python.client import device_lib
device_lib.list_local_devices()