import time
import tensorflow as tf
from src.networks.Theta import Theta

class MockModel:
    def __init__(self):
        self.nn_params = Theta([tf.zeros((10, 10))])
        self.eval_count = 0

    def grad_loss(self, data_batch, full_loss):
        self.eval_count += 1
        time.sleep(0.01) # Simulate expensive gradient evaluation
        return Theta([tf.ones((10, 10))])

class MockHMC:
    def __init__(self, L, dt, eta):
        self.model = MockModel()
        self.data_batch = None
        self.__full_loss = True
        self.HMC_L = L
        self.HMC_dt = dt
        self.HMC_eta = eta
        self.burn_in = 0
        self.curr_ep = 1

    def __leapfrog_step_old(self, old_theta, r, dt):
        grad_theta = self.model.grad_loss(self.data_batch, self.__full_loss)
        r = r - grad_theta * dt / 2
        self.model.nn_params = old_theta + r * dt
        grad_theta = self.model.grad_loss(self.data_batch, self.__full_loss)
        r = r - grad_theta * dt / 2
        return self.model.nn_params, r

    def sample_theta_old(self, theta_0):
        r_0 = theta_0.normal(self.HMC_eta)
        r   = r_0.copy()
        theta = theta_0.copy()
        for _ in range(self.HMC_L):
            theta, r = self.__leapfrog_step_old(theta, r, self.HMC_dt)
        return theta, r

    def __leapfrog_step_new(self, old_theta, r, dt, grad_theta):
        r = r - grad_theta * dt / 2
        self.model.nn_params = old_theta + r * dt
        grad_theta = self.model.grad_loss(self.data_batch, self.__full_loss)
        r = r - grad_theta * dt / 2
        return self.model.nn_params, r, grad_theta

    def sample_theta_new(self, theta_0):
        r_0 = theta_0.normal(self.HMC_eta)
        r   = r_0.copy()
        theta = theta_0.copy()

        self.model.nn_params = theta
        grad_theta = self.model.grad_loss(self.data_batch, self.__full_loss)

        for _ in range(self.HMC_L):
            theta, r, grad_theta = self.__leapfrog_step_new(theta, r, self.HMC_dt, grad_theta)
        return theta, r

hmc = MockHMC(L=10, dt=0.1, eta=1.0)
theta_0 = Theta([tf.zeros((10, 10))])

# Benchmarking old
hmc.model.eval_count = 0
t0 = time.time()
hmc.sample_theta_old(theta_0)
t1 = time.time()
print(f"Old: {t1-t0:.4f} s, Grad Evals: {hmc.model.eval_count}")

# Benchmarking new
hmc.model.eval_count = 0
t0 = time.time()
hmc.sample_theta_new(theta_0)
t1 = time.time()
print(f"New: {t1-t0:.4f} s, Grad Evals: {hmc.model.eval_count}")
