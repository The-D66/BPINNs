import time
import tensorflow as tf
from src.networks.Theta import Theta

class MockModel:
    def __init__(self):
        self.nn_params = None
        self.call_count = 0
    def grad_loss(self, data, full_loss):
        self.call_count += 1
        # dummy grad
        return self.nn_params * 0.1

class MockHMC:
    def __init__(self, L, dt):
        self.model = MockModel()
        self.HMC_L = L
        self.HMC_dt = dt
        self.data_batch = None
        self.__full_loss = True

    def __leapfrog_step_original(self, old_theta, r, dt):
        grad_theta = self.model.grad_loss(self.data_batch, self.__full_loss)
        r = r - grad_theta * dt / 2
        self.model.nn_params = old_theta + r * dt
        grad_theta = self.model.grad_loss(self.data_batch, self.__full_loss)
        r = r - grad_theta * dt / 2
        return self.model.nn_params, r

    def sample_original(self, theta_0, r_0):
        r = r_0.copy()
        theta = theta_0.copy()
        self.model.nn_params = theta
        for _ in range(self.HMC_L):
            theta, r = self.__leapfrog_step_original(theta, r, self.HMC_dt)
        return theta, r

    def __leapfrog_step_opt(self, old_theta, r, grad_theta, dt):
        r = r - grad_theta * dt / 2
        self.model.nn_params = old_theta + r * dt
        grad_theta = self.model.grad_loss(self.data_batch, self.__full_loss)
        r = r - grad_theta * dt / 2
        return self.model.nn_params, r, grad_theta

    def sample_opt(self, theta_0, r_0):
        r = r_0.copy()
        theta = theta_0.copy()
        self.model.nn_params = theta
        grad_theta = self.model.grad_loss(self.data_batch, self.__full_loss)
        for _ in range(self.HMC_L):
            theta, r, grad_theta = self.__leapfrog_step_opt(theta, r, grad_theta, self.HMC_dt)
        return theta, r

hmc = MockHMC(10, 0.1)
theta_0 = Theta([tf.ones((10, 10))])
r_0 = Theta([tf.ones((10, 10))])

hmc.model.call_count = 0
t, r = hmc.sample_original(theta_0, r_0)
print(f"Original: {hmc.model.call_count} calls, res sum: {t.ssum().numpy()}")

hmc.model.call_count = 0
t, r = hmc.sample_opt(theta_0, r_0)
print(f"Optimized: {hmc.model.call_count} calls, res sum: {t.ssum().numpy()}")
