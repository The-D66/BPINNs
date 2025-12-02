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

    def update_active_losses(self, losses_config):
        """ Override to sync history dict when active losses change """
        super().update_active_losses(losses_config)
        # Ensure all new metrics have entries in history
        for key in self.metric + ["Total"]:
            if key not in self.history[0]:
                self.history[0][key] = list()
                self.history[1][key] = list()
