import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from src.data import generate_synthetic_data, load_jena_climate, get_train_val_test_splits
from src.model import ProbabilisticForecaster
from src.train import create_train_state, make_epoch_step_fn, make_eval_step_fn
from src.evaluate import unscale_predictions, calculate_metrics
from src.visualize import plot_fan_chart, plot_calibration_curve

def run_project1(num_epochs=15, batch_size=32, lr=1e-3, seed=42, use_real_data=False):
    print("=" * 60)
    print("Starting Project 1: Probabilistic Time-Series Forecasting")
    print("=" * 60)
    
    # 1. Load data
    if use_real_data:
        print("Loading real Jena Climate dataset...")
        df = load_jena_climate(seed=seed)
    else:
        print("Generating synthetic store sales data...")
        df = generate_synthetic_data(num_series=12, num_days=365 * 3, seed=seed)
    
    lookback = 30
    horizon = 7
    quantiles = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_train_val_test_splits(
        df, lookback=lookback, horizon=horizon, val_ratio=0.15, test_ratio=0.15
    )
    
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Val samples:   {X_val.shape[0]}")
    print(f"Test samples:  {X_test.shape[0]}")
    print(f"Features shape: {X_train.shape[2]} (sales_scaled, day_of_week_scaled, month_scaled, is_event)")
    
    # 2. Batch the training data for jax.lax.scan training
    num_samples = X_train.shape[0]
    num_batches = num_samples // batch_size
    # Trim to fit integer number of batches
    X_train_batched = X_train[:num_batches * batch_size].reshape((num_batches, batch_size, lookback, X_train.shape[2]))
    y_train_batched = y_train[:num_batches * batch_size].reshape((num_batches, batch_size, horizon + 2))
    
    # 3. Initialize model and optimization state
    key = jax.random.PRNGKey(seed)
    model_key, init_key = jax.random.split(key)
    
    model = ProbabilisticForecaster(
        horizon=horizon,
        num_quantiles=len(quantiles),
        embed_dim=32,
        num_heads=2,
        head_dim=16,
        mlp_dim=64,
        num_layers=2
    )
    
    input_shape = (batch_size, lookback, X_train.shape[2])
    print("Initializing state and compiling model step functions...")
    state = create_train_state(model, init_key, lr, input_shape)
    
    train_epoch_fn = make_epoch_step_fn(model, quantiles, horizon)
    eval_epoch_fn = make_eval_step_fn(model, quantiles, horizon)
    
    # 4. Training loop
    print("Training Custom Linear Transformer on Multi-Quantile Pinball Loss...")
    for epoch in range(1, num_epochs + 1):
        # Run training epoch (highly optimized JIT loop via lax.scan)
        state, train_loss = train_epoch_fn(state, (X_train_batched, y_train_batched))
        
        # Validation evaluation
        val_loss, val_crps, _ = eval_epoch_fn(state, X_val, y_val)
        
        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"Epoch {epoch:02d} | Train Pinball+Crossing Loss: {train_loss:.4f} | Val Pinball Loss: {val_loss:.4f} | Val CRPS: {val_crps:.4f}")
            
    # 5. Final Evaluation on Test Set
    print("\nRunning final evaluation on unseen test set...")
    test_loss, test_crps, y_pred_scaled = eval_epoch_fn(state, X_test, y_test)
    
    # Unscale predictions and targets to actual sales space
    y_true_unscaled, y_pred_unscaled = unscale_predictions(
        np.array(y_pred_scaled), np.array(y_test), horizon
    )
    
    metrics = calculate_metrics(y_true_unscaled, y_pred_unscaled, np.array(quantiles))
    
    print("\n" + "-" * 40)
    print("Probabilistic Forecast Evaluation Summary:")
    print("-" * 40)
    print(f"Continuous Ranked Probability Score (CRPS): {test_crps:.4f}")
    if "point_metrics" in metrics:
        print(f"Point Predictor (Median) MAE:               {metrics['point_metrics']['MAE']:.4f}")
        print(f"Point Predictor (Median) RMSE:              {metrics['point_metrics']['RMSE']:.4f}")
    if "80%_interval" in metrics:
        print("\n80% Prediction Interval (Nominal Target: 0.80):")
        print(f"  Empirical Coverage (PICP):                 {metrics['80%_interval']['PICP']:.4f}")
        print(f"  Mean Interval Width (MPIW - Sharpness):     {metrics['80%_interval']['MPIW']:.4f}")
        print(f"  Winkler Score (Interval Penalty):          {metrics['80%_interval']['Winkler_Score']:.4f}")
    if "50%_interval" in metrics:
        print("\n50% Prediction Interval (Nominal Target: 0.50):")
        print(f"  Empirical Coverage (PICP):                 {metrics['50%_interval']['PICP']:.4f}")
        print(f"  Mean Interval Width (MPIW - Sharpness):     {metrics['50%_interval']['MPIW']:.4f}")
        print(f"  Winkler Score (Interval Penalty):          {metrics['50%_interval']['Winkler_Score']:.4f}")
    print(f"\nQuantile Crossing Rate:                     {metrics['quantile_crossing_rate']:.4%}")
    print("-" * 40)
    
    # 6. Save visualizations
    print("\nGenerating evaluation plots...")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    
    fan_chart_path = os.path.join(results_dir, "fan_chart.png")
    plot_fan_chart(y_true_unscaled, y_pred_unscaled, np.array(quantiles), series_idx=0, save_path=fan_chart_path)
    print(f"Saved fan chart to {fan_chart_path}")
    
    cal_path = os.path.join(results_dir, "calibration_reliability.png")
    plot_calibration_curve(y_true_unscaled, y_pred_unscaled, np.array(quantiles), save_path=cal_path)
    print(f"Saved calibration reliability curve to {cal_path}")
    
    print("\nProject 1 execution complete!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--real-data", action="store_true", help="Use real Jena Climate dataset instead of synthetic data")
    args = parser.parse_args()
    
    run_project1(num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, use_real_data=args.real_data)
