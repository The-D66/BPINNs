import time
import numpy as np
import tensorflow as tf
from src.networks.Theta import Theta

# Mock the model and data batch
class MockModel:
    def __init__(self):
        self.nn_params = Theta([tf.zeros((100, 100))])

    def grad_loss(self, data_batch, full_loss):
        # simulate some compute
        tf.matmul(self.nn_params.values[0], self.nn_params.values[0])
        return Theta([tf.ones((100, 100)) * 0.1])

class MockHMC:
    def __init__(self):
        self.model = MockModel()
        self.data_batch = None
        self.__full_loss = True
        self.HMC_L = 10
        self.HMC_dt = 0.01

    def __leapfrog_step_old(self, old_theta, r, dt):
        grad_theta = self.model.grad_loss(self.data_batch, self.__full_loss)
        r = r - grad_theta * dt / 2
        self.model.nn_params = old_theta + r * dt
        grad_theta = self.model.grad_loss(self.data_batch, self.__full_loss)
        r = r - grad_theta * dt / 2
        return self.model.nn_params, r

    def sample_theta_old(self, theta_0, r_0):
        r = r_0.copy()
        theta = theta_0.copy()
        for _ in range(self.HMC_L):
            theta, r = self.__leapfrog_step_old(theta, r, self.HMC_dt)
        return theta, r

    def __leapfrog_step_new(self, old_theta, r, dt, grad_theta):
        r = r - grad_theta * dt / 2
        self.model.nn_params = old_theta + r * dt
        grad_theta_new = self.model.grad_loss(self.data_batch, self.__full_loss)
        r = r - grad_theta_new * dt / 2
        return self.model.nn_params, r, grad_theta_new

    def sample_theta_new(self, theta_0, r_0):
        r = r_0.copy()
        theta = theta_0.copy()
        grad_theta = self.model.grad_loss(self.data_batch, self.__full_loss)
        for _ in range(self.HMC_L):
            theta, r, grad_theta = self.__leapfrog_step_new(theta, r, self.HMC_dt, grad_theta)
        return theta, r

# Test
hmc = MockHMC()
theta_0 = Theta([tf.zeros((100, 100))])
r_0 = Theta([tf.random.normal((100, 100))])

# Warmup
hmc.sample_theta_old(theta_0, r_0)
hmc.sample_theta_new(theta_0, r_0)

t0 = time.time()
for _ in range(100):
    hmc.sample_theta_old(theta_0, r_0)
t_old = time.time() - t0

t0 = time.time()
for _ in range(100):
    hmc.sample_theta_new(theta_0, r_0)
t_new = time.time() - t0

print(f"Old time: {t_old:.4f}s")
print(f"New time: {t_new:.4f}s")
print(f"Speedup: {t_old/t_new:.2f}x")
