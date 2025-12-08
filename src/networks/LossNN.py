from .PhysNN import PhysNN
from networks.Theta import Theta
from equations.Operators import Operators
import tensorflow as tf

class LossNN(PhysNN):
    """
    Evaluate PDEs residuals (using pde constraint)
    Compute mean-squared-errors and loglikelihood
        - residual loss (pdes)
        - boundary loss (boundary conditions)
        - data loss (fitting)
        - prior loss
    Losses structure
        - loss_total: tuple (mse, loglikelihood)
        - mse, loglikelihood: dictionaries with keys relative to loss type
    """

    def __init__(self, par, **kw):
        super(LossNN, self).__init__(par, **kw)
        self.metric = [k for k,v in par.metrics.items() if v]
        self.keys   = [k for k,v in  par.losses.items() if v]
        self.vars   = par.uncertainty

    @staticmethod
    def __mse(vect):
        """ Mean Squared Error """
        norm = tf.norm(vect, axis = -1)
        return tf.keras.losses.MSE(norm, tf.zeros_like(norm))

    @staticmethod
    def __normal_loglikelihood(mse, n, log_var):
        """ Negative log-likelihood """
        return 0.5 * n * ( mse * tf.math.exp(log_var) - log_var)

    def __loss_data(self, outputs, targets, log_var):
        """ Auxiliary loss function for the computation of Normal(output | target, 1 / beta * I) """
        post_data = self.__mse(outputs-targets)
        log_data = self.__normal_loglikelihood(post_data, outputs.shape[0], log_var)
        return self.tf_convert(post_data), self.tf_convert(log_data)

    def __loss_data_u(self, data):
        """ Fitting loss on u; computation of the residual at points of measurement of u """
        outputs = self.forward(data["dom"])
        log_var = tf.math.log(1/self.vars["sol"]**2)
        return self.__loss_data(outputs[0], data["sol"], log_var)

    def __loss_data_f(self, data):
        """ Fitting loss on f; computation of the residual at points of measurement of f """
        outputs = self.forward(data["dom"])
        log_var = tf.math.log(1/self.vars["par"]**2)
        return self.__loss_data(outputs[1], data["par"], log_var)

    def __loss_data_b(self, data):
        """ Boundary loss; computation of the residual on boundary conditions """
        outputs = self.forward(data["dom"])
        log_var = tf.math.log(1/self.vars["bnd"]**2)
        return self.__loss_data(outputs[0], data["sol"], log_var)

    def __loss_residual(self, data):
        """ Physical loss; computation of the residual of the PDE """
        inputs = self.tf_convert(data["dom"])
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            u, f = self.forward(inputs)
            
            # RLPI Integration
            extra_fields = {}
            reg_loss_val = 0.0
            
            if getattr(self, "policy_nn", None) is not None:
                # Unpack solutions for gradients
                sol_list = Operators.tf_unpack(u)
                h_norm, u_norm = sol_list[0], sol_list[1]
                
                grad_h = Operators.gradient_scalar(tape, h_norm, inputs)
                grad_u = Operators.gradient_scalar(tape, u_norm, inputs)
                
                # Extract spatial gradients (d/dx is col 0)
                grad_h_x = grad_h[:, 0:1]
                grad_u_x = grad_u[:, 0:1]
                
                # State: [x, t, h, u, |hx|, |ux|] -> Total 6 dims
                state = tf.concat([
                    inputs,
                    h_norm, u_norm,
                    tf.abs(grad_h_x), tf.abs(grad_u_x)
                ], axis=1)
                
                mu = self.policy_nn.forward(state)
                
                # Detach if required (Solver Phase)
                if getattr(self, "rl_detach_mu", False):
                    mu_used = tf.stop_gradient(mu)
                else:
                    mu_used = mu
                    
                extra_fields["mu"] = mu_used
                
                # Regularization (L1 Penalty on mu)
                # Scaled by N to match log_res scaling
                reg_loss_val = self.tf_convert(inputs.shape[0]) * getattr(self, "lambda_reg", 0.0) * tf.reduce_mean(tf.abs(mu))

            residuals = self.pinn.comp_residual(inputs, u, f, tape, extra_fields=extra_fields)
        
        # Causal Training: Weight residuals by exp(-lambda * t)
        t_norm = inputs[:, 1:2]
        lambda_causal = 5.0
        causal_weight = tf.exp(-0.5 * lambda_causal * t_norm)
        residuals = residuals * causal_weight

        mse = self.__mse(residuals)
        log_var =  tf.math.log(1/self.vars["pde"]**2)
        log_res = self.__normal_loglikelihood(mse, inputs.shape[0], log_var)
        
        # Add regularization to the optimization target
        log_res = log_res + self.tf_convert(reg_loss_val)
        
        return mse, log_res

    def __loss_prior(self):
        """ Prior for neural network parameters, assuming them to be distributed as a gaussian N(0,stddev^2) """
        theta = Theta(self.model.trainable_variables)
        log_var = tf.math.log(1/self.stddev**2)
        prior   = theta.ssum()/theta.size()
        loglike = self.__normal_loglikelihood(prior, theta.size(), log_var)
        return prior, loglike

    def __compute_loss(self, dataset, keys, full_loss = True):
        """ Computation of the losses listed keys """
        pst, llk = dict(), dict()
        if "data_u" in keys: pst["data_u"], llk["data_u"] = self.__loss_data_u(dataset.data_sol)
        if "data_f" in keys: pst["data_f"], llk["data_f"] = self.__loss_data_f(dataset.data_par)
        if "data_b" in keys: pst["data_b"], llk["data_b"] = self.__loss_data_b(dataset.data_bnd)
        if "prior"  in keys: pst["prior"],  llk["prior"]  = self.__loss_prior()
        if "pde"    in keys: 
            pst["pde"], llk["pde"] = self.__loss_residual(dataset.data_pde) if full_loss else (self.tf_convert(0.0), self.tf_convert(0.0))
            # Debug print for PDE Loss
            # if full_loss:
            #     tf.print(f"DEBUG: PDE MSE in __compute_loss: {pst['pde']}", summarize=-1)
        return pst, llk

    def metric_total(self, dataset, full_loss = True):
        """ Computation of the losses required to be tracked """
        pst, llk = self.__compute_loss(dataset, self.metric, full_loss)
        pst["Total"] = sum(pst.values())
        llk["Total"] = sum(llk.values())
        return pst, llk

    def loss_total(self, dataset, full_loss = True):
        """ Creation of the dictionary containing all posteriors and log-likelihoods """
        _, llk = self.__compute_loss(dataset, self.keys, full_loss)
        return sum(llk.values())

    def update_active_losses(self, losses_config):
        """ Update the list of active losses based on a new configuration dictionary """
        # Handle both boolean strings (from JSON) and booleans
        def to_bool(v):
            if isinstance(v, bool): return v
            if isinstance(v, str): return v.lower() == "true"
            return False

        self.keys = [k for k,v in losses_config.items() if to_bool(v)]
        self.metric = self.keys # Keep metrics consistent with active losses for correct logging

    def grad_loss(self, dataset, full_loss = True, variables = None):
        """ Computation of the gradient of the loss function with respect to the network trainable parameters """
        if variables is None:
            variables = self.model.trainable_variables
            
        with tf.GradientTape(persistent=True) as tape:
            # tape.watch(variables) # Removed: Variables are watched automatically
            diff_llk = self.loss_total(dataset, full_loss)
        grad_thetas = tape.gradient(diff_llk, variables)
        return Theta(grad_thetas)

