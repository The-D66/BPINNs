import time
import numpy as np
import tensorflow as tf
from src.networks.Theta import Theta
from src.algorithms.HMC import HMC

class MockModel:
    def __init__(self):
        self.nn_params = Theta([tf.zeros((10, 10))])

    def grad_loss(self, data_batch, full_loss):
        # simulate some compute that takes time, to mimic actual training
        tf.matmul(self.nn_params.values[0], self.nn_params.values[0])
        time.sleep(0.005) # Add small sleep to make grad_loss dominate
        return Theta([tf.ones((10, 10)) * 0.1])

class MockHMC(HMC):
    def __init__(self):
        self.model = MockModel()
        self.data_batch = None
        self.__full_loss = True
        self.HMC_L = 10
        self.HMC_dt = 0.01
        self.burn_in = 0
        self.curr_ep = 1
        self.HMC_eta = 0.1
        self.debug_flag = True # to avoid epochs_loop
        self.selected = []

    def _HMC__hamiltonian(self, theta, r):
        return 0.5

# Test
hmc = MockHMC()
theta_0 = Theta([tf.zeros((10, 10))])

# Warmup
hmc.sample_theta(theta_0)

t0 = time.time()
for _ in range(10):
    hmc.sample_theta(theta_0)
t_new = time.time() - t0

print(f"New time: {t_new:.4f}s")
