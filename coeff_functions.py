

class CoeffFunctions:
    """ A set of functions describing a second-order PDE """
    def __init__(self, rhs_func, stiff_coeff_func, mass_coeff_func, dirichlet_func):
        self.rhs = rhs_func
        self.stiffness = stiff_coeff_func
        self.mass = mass_coeff_func
        self.dirichlet = dirichlet_func
