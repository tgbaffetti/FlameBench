from .AE import AE


class VAE(AE):
    """Beta-VAE. Evaluation uses the posterior mean for reproducible forecasts."""
    name = "vae"
    def __init__(self, beta=1e-4, **kwargs):
        if beta <= 0:
            raise ValueError("VAE beta must be positive")
        super().__init__(beta=beta, **kwargs)
