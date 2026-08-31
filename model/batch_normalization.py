import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists

        # Convert input data and learnable parameters to NumPy arrays
        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)

        # Convert running statistics to float64 NumPy arrays
        # These values are maintained across batches during training
        running_mean = np.array(running_mean, dtype=np.float64)
        running_var = np.array(running_var, dtype=np.float64)

        if training:
            # Compute the mean of each feature across the current batch
            # axis=0 means calculating statistics column by column
            batch_mean = np.mean(x, axis=0)

            # Compute the variance of each feature across the current batch
            batch_var = np.var(x, axis=0)

            # Normalize the input using the current batch statistics
            # eps prevents division by zero or numerical instability
            x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)

            # Update the running mean using exponential moving average
            running_mean = (
                (1 - momentum) * running_mean
                + momentum * batch_mean
            )

            # Update the running variance using exponential moving average
            running_var = (
                (1 - momentum) * running_var
                + momentum * batch_var
            )

        else:
            # During inference, use the stored running statistics
            # instead of statistics from the current batch
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)

        # Apply the learnable affine transformation:
        # y = gamma * normalized_x + beta
        out = gamma * x_hat + beta

        # Round all results to 4 decimal places and convert
        # NumPy arrays back to regular Python lists
        return (
            np.round(out, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )

       
