import matplotlib.pyplot as plt
import numpy as np
import os

def setup_dark_theme():
    """Sets a clean, modern dark style for matplotlib plots."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'grid.color': '#2A2A2A',
        'grid.linestyle': '--',
        'axes.facecolor': '#1E1E1E',
        'figure.facecolor': '#121212',
        'text.color': '#E0E0E0',
        'axes.labelcolor': '#CCCCCC',
        'xtick.color': '#888888',
        'ytick.color': '#888888',
        'font.family': 'sans-serif',
        'font.size': 10,
        'figure.autolayout': True
    })

def plot_fan_chart(y_true, y_pred, quantiles, series_idx=0, step_limit=100, save_path=None):
    """
    Plots a probabilistic forecast fan chart showing historical actuals and future predicted bands.
    """
    setup_dark_theme()
    
    # Select sample data
    # y_true: (num_samples, horizon) -> take the first series_idx
    actuals = y_true[series_idx]
    preds = y_pred[series_idx] # (horizon, num_quantiles)
    
    q_indices = {q: idx for idx, q in enumerate(quantiles)}
    
    horizon = len(actuals)
    x = np.arange(horizon)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot true values
    ax.plot(x, actuals, color="#FF9F43", label="Actual Sales", linewidth=2.5, marker='o', markersize=4)
    
    # Plot predicted median
    if 0.5 in q_indices:
        ax.plot(x, preds[:, q_indices[0.5]], color="#00D2D3", label="Predicted Median", linewidth=2, linestyle='--')
        
    # Shaded band for 50% interval (0.25 to 0.75)
    if 0.25 in q_indices and 0.75 in q_indices:
        ax.fill_between(
            x, 
            preds[:, q_indices[0.25]], 
            preds[:, q_indices[0.75]], 
            color="#00D2D3", 
            alpha=0.3, 
            label="50% Prediction Interval"
        )
        
    # Shaded band for 80% interval (0.10 to 0.90)
    if 0.1 in q_indices and 0.9 in q_indices:
        ax.fill_between(
            x, 
            preds[:, q_indices[0.1]], 
            preds[:, q_indices[0.9]], 
            color="#00D2D3", 
            alpha=0.15, 
            label="80% Prediction Interval"
        )
        
    # Styling
    ax.set_title(f"Probabilistic Demand Forecast (Sample Store Series {series_idx})", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Forecast Horizon Step", fontsize=10)
    ax.set_ylabel("Sales Volume", fontsize=10)
    ax.grid(True)
    ax.legend(loc="upper left", framealpha=0.8, facecolor="#1A1A1A", edgecolor="#333333")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_calibration_curve(y_true, y_pred, quantiles, save_path=None):
    """
    Plots nominal vs empirical coverage to evaluate how well calibrated the forecast intervals are.
    """
    setup_dark_theme()
    
    empirical_coverages = []
    nominal_coverages = []
    
    # For a range of nominal coverages: e.g. 10%, 20%, ..., 90%
    # We can evaluate the coverage of symmetric intervals
    # e.g. 50% interval: [q_0.25, q_0.75], 80% interval: [q_0.10, q_0.90]
    intervals = [
        (0.25, 0.75, 0.50),
        (0.10, 0.90, 0.80),
    ]
    
    # We can also evaluate individual quantiles (e.g. check if y <= q_val has fraction <= q_val)
    # This is more granular and tests every quantile.
    for q_val in quantiles:
        q_idx = np.where(np.isclose(quantiles, q_val))[0][0]
        # Check fraction of times actual is below the predicted quantile
        emp_cov = np.mean(y_true <= y_pred[..., q_idx])
        empirical_coverages.append(emp_cov)
        nominal_coverages.append(q_val)
        
    # Sort
    sort_idx = np.argsort(nominal_coverages)
    nom = np.array(nominal_coverages)[sort_idx]
    emp = np.array(empirical_coverages)[sort_idx]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", label="Perfect Calibration")
    
    # Model calibration line
    ax.plot(nom, emp, marker='o', markersize=6, color="#54A0FF", linewidth=2.5, label="Linear Transformer")
    
    # Scatter points color-coded or highlighted
    ax.scatter(nom, emp, color="#10AC84", zorder=5)
    
    # Styling
    ax.set_title("Quantile Calibration Reliability Diagram", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Nominal Quantile Level", fontsize=10)
    ax.set_ylabel("Empirical Coverage Fraction", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.legend(loc="upper left", framealpha=0.8, facecolor="#1A1A1A", edgecolor="#333333")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
