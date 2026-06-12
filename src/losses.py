import jax.numpy as jnp

def pinball_loss(y, y_pred, quantiles):
    """
    Computes the pinball loss (quantile loss) for multiple quantiles.
    Args:
        y: target values of shape (batch_size, horizon)
        y_pred: predicted quantiles of shape (batch_size, horizon, num_quantiles)
        quantiles: list/array of quantile levels of shape (num_quantiles,)
    """
    # Expand y to match y_pred shape: (batch_size, horizon, 1)
    y_expanded = jnp.expand_dims(y, axis=-1)
    
    # Error: (batch_size, horizon, num_quantiles)
    error = y_expanded - y_pred
    
    # Broadcast quantiles to (1, 1, num_quantiles)
    q = jnp.reshape(quantiles, (1, 1, -1))
    
    # Pinball loss: max(q * error, (q - 1) * error)
    loss = jnp.maximum(q * error, (q - 1) * error)
    
    # Average over batch, horizon, and quantiles
    return jnp.mean(loss)

def quantile_crossing_penalty(y_pred, penalty_weight=1.0):
    """
    Penalizes when quantiles cross (i.e. lower quantile prediction is larger than higher quantile prediction).
    y_pred shape: (batch_size, horizon, num_quantiles)
    """
    # Compute differences between consecutive quantiles along the last axis
    diffs = jnp.diff(y_pred, axis=-1) # shape: (batch_size, horizon, num_quantiles - 1)
    
    # Penalty: max(0, -diff)
    penalty = jnp.maximum(0.0, -diffs)
    
    return penalty_weight * jnp.mean(penalty)

def total_probabilistic_loss(y, y_pred, quantiles, penalty_weight=10.0):
    """
    Combines pinball loss and quantile crossing penalty.
    """
    loss_pinball = pinball_loss(y, y_pred, quantiles)
    loss_crossing = quantile_crossing_penalty(y_pred, penalty_weight)
    return loss_pinball + loss_crossing

def approximate_crps(y, y_pred, quantiles):
    """
    Approximates CRPS using the relationship: CRPS(y, F) = 2 * E_q[Pinball(y, q)]
    y shape: (batch_size, horizon)
    y_pred shape: (batch_size, horizon, num_quantiles)
    """
    # Expand y: (batch_size, horizon, 1)
    y_expanded = jnp.expand_dims(y, axis=-1)
    error = y_expanded - y_pred
    q = jnp.reshape(quantiles, (1, 1, -1))
    
    # Pinball loss per quantile and sample
    pinball = jnp.maximum(q * error, (q - 1) * error) # (batch_size, horizon, num_quantiles)
    
    # Average across quantiles, then average across batch and horizon, multiply by 2
    mean_pinball_per_step = jnp.mean(pinball, axis=-1) # (batch_size, horizon)
    crps = 2.0 * jnp.mean(mean_pinball_per_step)
    return crps
