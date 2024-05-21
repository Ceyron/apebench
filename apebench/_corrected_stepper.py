import equinox as eqx
import jax


class CorrectedStepper(eqx.Module):
    """
    Take a given coarse stepper (whose parameters are fixed) and a network that
    is supposed to correct it.
    """

    coarse_stepper: eqx.Module
    network: eqx.Module
    mode: str

    def __call__(
        self,
        x,
    ):
        # Detach the coarse solver because it is not supposed to be trained
        coarse_solver_detached = jax.lax.stop_gradient(self.coarse_stepper)

        if self.mode == "parallel":
            network_prediction = self.network(x)

            coarse_solver_prediction = coarse_solver_detached(x)

            next_x = coarse_solver_prediction + network_prediction
        elif self.mode == "sequential":
            coarse_solver_prediction = coarse_solver_detached(x)

            network_prediction = self.network(coarse_solver_prediction)

            next_x = network_prediction
        elif self.mode == "sequential_with_bypass":
            coarse_solver_prediction = coarse_solver_detached(x)

            network_prediction = self.network(coarse_solver_prediction)

            next_x = coarse_solver_prediction + network_prediction
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        return next_x
