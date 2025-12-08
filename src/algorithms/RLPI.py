from algorithms.Algorithm import Algorithm
import tensorflow as tf

class RLPI(Algorithm):
    """
    Class for RLPI (Reinforcement Learning Physics Informed) training.
    Alternates between optimizing Solver and Policy.
    """
    def __init__(self, bayes_nn, param_method, debug_flag):
        super().__init__(bayes_nn, param_method, debug_flag)
        
        self.lr      = param_method.get("lr", 1e-3)
        self.beta_1  = param_method.get("beta_1", 0.9)
        self.beta_2  = param_method.get("beta_2", 0.999)
        self.eps     = param_method.get("eps", 1e-7)
        
        self.rl_warmup = param_method.get("rl_warmup", 1000)
        
        # Initialize Momentum for Solver
        self.m_solver = self.model.nn_params * 0
        self.v_solver = self.model.nn_params * 0
        
        # Initialize Momentum for Policy
        if getattr(self.model, "policy_nn", None) is not None:
            self.m_policy = self.model.policy_nn.nn_params * 0
            self.v_policy = self.model.policy_nn.nn_params * 0
        else:
            print("Warning: RLPI selected but no PolicyNN found in model.")
        
        # Training Schedule (5:1 ratio)
        self.solver_steps = 5
        self.policy_steps = 1

    def _adam_update(self, theta, grad, m, v):
        m = m * self.beta_1 + grad * (1 - self.beta_1)
        v = v * self.beta_2 + (grad**2) * (1 - self.beta_2)
        
        # Bias correction (simplified/implicit as per ADAM.py)
        # theta -= (m / (1-beta1) * lr) / (sqrt(v / (1-beta2)) + eps)
        
        step = (m / (1 - self.beta_1) * self.lr) / ((v / (1 - self.beta_2))**0.5 + self.eps)
        return theta - step, m, v

    def sample_theta(self, theta_0):
        full_loss = True 
        
        # Determine Phase
        # Warmup: Train Solver Only
        if self.curr_ep <= self.rl_warmup:
            self.model.rl_detach_mu = True 
            
            # Gradient for Solver
            grad_solver = self.model.grad_loss(self.data_batch, full_loss, variables=self.model.model.trainable_variables)
            
            # Update Solver
            new_theta, self.m_solver, self.v_solver = self._adam_update(theta_0, grad_solver, self.m_solver, self.v_solver)
            
            return new_theta

        else:
            # Joint Training Phase
            cycle = self.solver_steps + self.policy_steps
            # Relative epoch index after warmup
            rel_ep = self.curr_ep - self.rl_warmup - 1
            pos = rel_ep % cycle
            
            if pos < self.solver_steps:
                # Solver Update Phase
                self.model.rl_detach_mu = True
                
                grad_solver = self.model.grad_loss(self.data_batch, full_loss, variables=self.model.model.trainable_variables)
                new_theta, self.m_solver, self.v_solver = self._adam_update(theta_0, grad_solver, self.m_solver, self.v_solver)
                
                return new_theta
            else:
                # Policy Update Phase
                self.model.rl_detach_mu = False # Retain graph to backprop through mu
                
                if getattr(self.model, "policy_nn", None) is None:
                    return theta_0 # Should not happen
                
                policy_theta = self.model.policy_nn.nn_params
                
                grad_policy = self.model.grad_loss(self.data_batch, full_loss, variables=self.model.policy_nn.model.trainable_variables)
                
                new_policy_theta, self.m_policy, self.v_policy = self._adam_update(policy_theta, grad_policy, self.m_policy, self.v_policy)
                
                # Update Policy weights directly
                self.model.policy_nn.nn_params = new_policy_theta
                
                # Return Solver weights (unchanged)
                return theta_0

    def select_thetas(self, thetas_train):
        # Just return the last one, similar to ADAM
        return thetas_train[-1:]
