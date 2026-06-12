import numpy as np

def unscale_predictions(y_pred_scaled, y_scaled_with_info, horizon):
    """
    Unscales predicted quantiles and targets using the scaling factors (mean, std)
    stored inside the target array.
    Args:
        y_pred_scaled: predicted quantiles of shape (num_samples, horizon, num_quantiles)
        y_scaled_with_info: target array of shape (num_samples, horizon + 2)
    Returns:
        y_true_unscaled: (num_samples, horizon)
        y_pred_unscaled: (num_samples, horizon, num_quantiles)
    """
    # Extract actuals, means, stds
    y_true_scaled = y_scaled_with_info[:, :horizon]
    means = y_scaled_with_info[:, horizon][:, np.newaxis] # (num_samples, 1)
    stds = y_scaled_with_info[:, horizon + 1][:, np.newaxis] # (num_samples, 1)
    
    # Unscale actuals
    y_true_unscaled = y_true_scaled * stds + means
    
    # Unscale predictions (broadcast along quantiles dim)
    means_expanded = np.expand_dims(means, axis=-1) # (num_samples, 1, 1)
    stds_expanded = np.expand_dims(stds, axis=-1)   # (num_samples, 1, 1)
    y_pred_unscaled = y_pred_scaled * stds_expanded + means_expanded
    
    return y_true_unscaled, y_pred_unscaled

def calculate_metrics(y_true, y_pred, quantiles):
    """
    Computes PICP, MPIW, Winkler score, and CRPS for nominal coverage levels.
    Args:
        y_true: unscaled actuals (num_samples, horizon)
        y_pred: unscaled predictions (num_samples, horizon, num_quantiles)
        quantiles: list of quantiles used in the predictions
    """
    metrics = {}
    
    # We want to evaluate the 80% prediction interval (q=0.1 to q=0.9)
    # and 50% prediction interval (q=0.25 to q=0.75) if available
    q_indices = {q: idx for idx, q in enumerate(quantiles)}
    
    # 80% Interval
    if 0.1 in q_indices and 0.9 in q_indices:
        idx_low, idx_high = q_indices[0.1], q_indices[0.9]
        metrics["80%_interval"] = evaluate_interval(
            y_true, y_pred[..., idx_low], y_pred[..., idx_high], alpha=0.2
        )
        
    # 50% Interval
    if 0.25 in q_indices and 0.75 in q_indices:
        idx_low, idx_high = q_indices[0.25], q_indices[0.75]
        metrics["50%_interval"] = evaluate_interval(
            y_true, y_pred[..., idx_low], y_pred[..., idx_high], alpha=0.5
        )
        
    # Point prediction error (using median q=0.5)
    if 0.5 in q_indices:
        median_pred = y_pred[..., q_indices[0.5]]
        mae = np.mean(np.abs(y_true - median_pred))
        rmse = np.sqrt(np.mean((y_true - median_pred) ** 2))
        metrics["point_metrics"] = {"MAE": float(mae), "RMSE": float(rmse)}
        
    # Quantile crossing rate
    diffs = np.diff(y_pred, axis=-1)
    crossing_rate = np.mean(diffs < 0)
    metrics["quantile_crossing_rate"] = float(crossing_rate)
    
    return metrics

def evaluate_interval(y_true, lower_bound, upper_bound, alpha):
    """
    Computes PICP, MPIW, and Winkler Score for a single nominal interval.
    """
    # Prediction Interval Coverage Probability (PICP)
    coverage = (y_true >= lower_bound) & (y_true <= upper_bound)
    picp = np.mean(coverage)
    
    # Mean Prediction Interval Width (MPIW)
    width = upper_bound - lower_bound
    mpiw = np.mean(width)
    
    # Winkler Score
    # winkler = width + (2/alpha) * (lower - y) if y < lower
    #           + (2/alpha) * (y - upper) if y > upper
    winkler = width.copy()
    below_mask = y_true < lower_bound
    above_mask = y_true > upper_bound
    
    winkler[below_mask] += (2.0 / alpha) * (lower_bound[below_mask] - y_true[below_mask])
    winkler[above_mask] += (2.0 / alpha) * (y_true[above_mask] - upper_bound[above_mask])
    winkler_score = np.mean(winkler)
    
    return {
        "PICP": float(picp),
        "MPIW": float(mpiw),
        "Winkler_Score": float(winkler_score)
    }
