import numpy as np
from scipy.integrate import solve_ivp
from GravModels.Models.Pines import Pines
from GravModels.Models.Mascon import Mascon
from GravModels.Models.Polyhedral import Polyhedral
from GravModels.Models.PointMass import PointMass


class Propagator:
    def __init__(
        self,
        asteroid,
        initial_position,
        initial_velocity,
        t_span,
        t_eval,
        model_type="Pines",
    ):
        self.asteroid = asteroid
        self.initial_conditions = np.hstack((initial_position, initial_velocity))
        self.t_span = t_span
        self.t_eval = t_eval
        self.model_type = model_type
        self.model = self._get_model()

    def _get_model(self):
        if self.model_type == "Pines":
            return Pines(self.asteroid)
        elif self.model_type == "Mascon":
            return Mascon(self.asteroid)
        elif self.model_type == "Poly":
            return Polyhedral(self.asteroid)
        elif self.model_type == "PointMass":
            return PointMass(self.asteroid)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def equations_of_motion(self, t, y):
        pos = y[:3]
        vel = y[3:]
        acc = self.model.calculate_acceleration(np.array([pos]))
        if acc.ndim == 2:
            acc = acc[0]
        dydt = np.hstack((vel, acc))
        return dydt

    def propagate(self):
        solution = solve_ivp(
            self.equations_of_motion,
            self.t_span,
            self.initial_conditions,
            t_eval=self.t_eval,
            method="DOP853",
            atol=1e-12,  # Absolute tolerance
            rtol=1e-12,  # Relative tolerance
        )
        return solution
