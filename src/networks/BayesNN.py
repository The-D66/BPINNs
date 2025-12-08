from .PredNN import PredNN
from .LossNN import LossNN

class BayesNN(PredNN, LossNN):
    """
    - Initialization of losses dictionaries and equation
    - Contains loss history
    """

    def __init__(self, par, equation):
        
        super(BayesNN, self).__init__(par = par, equation = equation)
        self.seed    = par.utils["random_seed"]
        self.history = self.__initialize_losses()
        self.constructors = (par, equation)
        
        # RLPI: Initialize Policy Network if selected
        if par.method == "RLPI":
            from networks.PolicyNN import PolicyNN
            self.policy_nn = PolicyNN(par)
            self.lambda_reg = par.utils.get("lambda_reg", 0.0)
            self.rl_detach_mu = False # Default: do not detach
        else:
            self.policy_nn = None

    def __initialize_losses(self):
        """ Initializes empty MSE and log-likelihood dictionaries """
        pst, llk = dict(), dict()
        for key in self.metric + ["Total"]: 
            pst[key] = list()
            llk[key] = list()
        return (pst, llk)

    def loss_step(self, new_losses):
        """ Appends new losses to loss history """
        for key in self.metric + ["Total"]: 
            self.history[0][key].append(new_losses[0][key])
            self.history[1][key].append(new_losses[1][key])
