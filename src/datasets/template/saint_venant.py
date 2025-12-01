from dataclasses import dataclass
from .template import Data_Config

@dataclass
class SaintVenant1D(Data_Config):
    name: str    = "SaintVenant1D"
    problem: str = "SaintVenant1D"
    pde: str     = "SaintVenant"
    # Physical domain dictionary
    phys_dom: dict = None
    # Computational domain dictionary (normalized)
    comp_dom: dict = None
    # Physics parameters
    physics = None
    # Inverse problem flag
    inv_flag: bool = False
    
    def __post_init__(self):
        if self.phys_dom is None:
            self.phys_dom = {
                "n_input": 2,    # x, t
                "n_out_sol": 2,  # h, u
                "n_out_par": 0   # No parameters to infer for now
            }
        if self.comp_dom is None:
            self.comp_dom = self.phys_dom
