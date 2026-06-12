import numpy as np
import pandas as pd

def generate_synthetic_data(num_series=10, num_days=365 * 3, seed=42):
    """
    Generates synthetic daily store sales data with trends, weekly seasonality,
    annual seasonality, calendar events, and autoregressive noise.
    """
    np.random.seed(seed)
    
    dates = pd.date_range(start="2023-01-01", periods=num_days, freq="D")
    all_data = []
    
    for s in range(num_series):
        # Base demand and linear trend
        base = np.random.uniform(50, 200)
        trend = np.linspace(0, np.random.uniform(10, 50), num_days)
        
        # Weekly seasonality (period = 7)
        weekly_pattern = np.array([0.8, 0.9, 1.0, 1.0, 1.1, 1.3, 1.2]) # higher on weekends
        weekly = weekly_pattern[dates.dayofweek] * base * 0.15
        
        # Annual seasonality (period = 365.25)
        day_of_year = dates.dayofyear
        annual = np.sin(2 * np.pi * day_of_year / 365.25) * base * 0.2
        
        # Calendar events (random holidays/promotions)
        events = np.zeros(num_days)
        event_days = np.random.choice(num_days, size=int(num_days * 0.05), replace=False)
        events[event_days] = np.random.uniform(1.5, 3.0, size=len(event_days)) * base * 0.3
        
        # Autoregressive component + noise (using Student-t noise for heavy tails)
        noise = np.zeros(num_days)
        ar_coef = 0.5
        # Student-t noise with df=4 for heavy tails
        raw_noise = np.random.standard_t(df=4, size=num_days) * (base * 0.08)
        for t in range(1, num_days):
            noise[t] = ar_coef * noise[t-1] + raw_noise[t]
            
        # Combine components
        sales = base + trend + weekly + annual + events + noise
        sales = np.clip(sales, a_min=1.0, a_max=None) # sales must be positive
        
        df = pd.DataFrame({
            "date": dates,
            "series_id": f"store_{s}",
            "sales": sales,
            "base_demand": base,
            "day_of_week": dates.dayofweek,
            "month": dates.month,
            "day_of_year": day_of_year,
            "is_event": (events > 0).astype(float)
        })
        all_data.append(df)
        
    return pd.concat(all_data, ignore_index=True)

def load_jena_climate(resample_freq='6h', seed=42):
    """
    Loads the Jena Climate dataset (real weather time-series data).
    Source: Max Planck Institute for Biogeochemistry.
    Features: Temperature, Pressure, Humidity, Wind, etc.
    Resolution: Originally 10-min intervals, resampled to 6-hour blocks.
    """
    import urllib.request
    import zipfile
    import os

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    os.makedirs(cache_dir, exist_ok=True)
    csv_path = os.path.join(cache_dir, 'jena_climate_2009_2016.csv')

    if not os.path.exists(csv_path):
        print("Downloading Jena Climate dataset...")
        url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
        zip_path = os.path.join(cache_dir, 'jena_climate.zip')
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(cache_dir)
        os.remove(zip_path)
        print("Download complete.")

    df = pd.read_csv(csv_path)
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    df = df.set_index('Date Time')

    # Resample to reduce granularity (10-min -> 6-hour means)
    df = df.resample(resample_freq).mean().dropna()

    # We'll forecast 'T (degC)' (temperature) as the target series
    # Create multiple "series" from different weather variables for multi-series forecasting
    # We'll treat each column as a separate univariate series
    target_col = 'T (degC)'

    # Build a DataFrame matching the synthetic data format
    all_data = []
    # Use temperature and 3 other key variables as separate series
    series_cols = ['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)']

    for i, col in enumerate(series_cols):
        series_df = pd.DataFrame({
            'date': df.index,
            'series_id': f'weather_{col.split(" ")[0]}',
            'sales': df[col].values,  # reuse 'sales' column name to match the existing contract
            'day_of_week': df.index.dayofweek,
            'month': df.index.month,
            'day_of_year': df.index.dayofyear,
            'is_event': 0.0  # no events in weather data
        })
        all_data.append(series_df)

    result = pd.concat(all_data, ignore_index=True)
    return result

def create_sliding_windows(df, lookback=30, horizon=7):
    """
    Creates lookback windows and targets for all time series.
    Returns:
        X: features of shape (num_samples, lookback, num_features)
        y: target values of shape (num_samples, horizon)
    """
    X_list = []
    y_list = []
    
    # Scale numerical features (e.g. sales) per series to keep it stable
    series_ids = df["series_id"].unique()
    
    for s_id in series_ids:
        series_df = df[df["series_id"] == s_id].sort_values("date").copy()
        
        # Scale sales locally
        sales = series_df["sales"].values
        mean_sales = sales.mean()
        std_sales = sales.std() + 1e-8
        scaled_sales = (sales - mean_sales) / std_sales
        
        # Create features
        features = np.stack([
            scaled_sales,
            (series_df["day_of_week"].values - 3.0) / 2.0, # normalized day of week
            (series_df["month"].values - 6.5) / 3.5,       # normalized month
            series_df["is_event"].values
        ], axis=1) # (num_days, num_features)
        
        num_days = len(series_df)
        for i in range(num_days - lookback - horizon + 1):
            x_win = features[i : i + lookback]
            # Target is the raw (unscaled) sales so that the loss function and metrics
            # predict the actual sales, but to make optimization easier we can also scale the target
            # and unscale it later, or pass the scaling factors.
            # Let's predict the scaled sales, and we will keep track of mean & std to unscale for evaluation.
            y_win = scaled_sales[i + lookback : i + lookback + horizon]
            
            # We also store mean & std so we can unscale the predictions
            # Let's package scale factors in the features or keep them separately
            # We can append mean and std as features to the lookback window or just return them
            scale_info = np.array([mean_sales, std_sales])
            
            # Let's add scale_info as part of the features or return them.
            # Let's append scale info to x_win so the model knows the scale, or keep it.
            # Actually, standardizing input is enough; the target being scaled is fine.
            X_list.append(x_win)
            y_list.append(np.concatenate([y_win, scale_info])) # target + mean + std (horizon + 2)
            
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

def get_train_val_test_splits(df, lookback=30, horizon=7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits the data temporally into train, validation, and test sets.
    """
    series_ids = df["series_id"].unique()
    num_days = len(df[df["series_id"] == series_ids[0]])
    
    test_days = int(num_days * test_ratio)
    val_days = int(num_days * val_ratio)
    train_days = num_days - test_days - val_days
    
    # Split the original DataFrame
    dates = df[df["series_id"] == series_ids[0]].sort_values("date")["date"].values
    train_end_date = dates[train_days]
    val_end_date = dates[train_days + val_days]
    
    train_df = df[df["date"] < train_end_date]
    val_df = df[(df["date"] >= train_end_date) & (df["date"] < val_end_date)]
    test_df = df[df["date"] >= val_end_date]
    
    X_train, y_train = create_sliding_windows(train_df, lookback, horizon)
    X_val, y_val = create_sliding_windows(val_df, lookback, horizon)
    X_test, y_test = create_sliding_windows(test_df, lookback, horizon)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_jena_climate(resample_freq='6h', seed=42):
    """
    Loads the Jena Climate dataset (real weather time-series data).
    Source: Max Planck Institute for Biogeochemistry.
    Features: Temperature, Pressure, Humidity, Wind, etc.
    Resolution: Originally 10-min intervals, resampled to 6-hour blocks.
    """
    import urllib.request
    import zipfile
    import os
    import pandas as pd
    import numpy as np
    
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    os.makedirs(cache_dir, exist_ok=True)
    csv_path = os.path.join(cache_dir, 'jena_climate_2009_2016.csv')
    
    if not os.path.exists(csv_path):
        print("Downloading Jena Climate dataset...")
        url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
        zip_path = os.path.join(cache_dir, 'jena_climate.zip')
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(cache_dir)
        os.remove(zip_path)
        print("Download complete.")
    
    df = pd.read_csv(csv_path)
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    df = df.set_index('Date Time')
    
    # Resample to reduce granularity (10-min -> 6-hour means)
    df = df.resample(resample_freq).mean().dropna()
    
    # We'll forecast 'T (degC)' (temperature) as the target series
    # Create multiple "series" from different weather variables for multi-series forecasting
    # We'll treat each column as a separate univariate series
    
    # Build a DataFrame matching the synthetic data format
    all_data = []
    # Use temperature and 3 other key variables as separate series
    series_cols = ['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)']
    
    for i, col in enumerate(series_cols):
        series_df = pd.DataFrame({
            'date': df.index,
            'series_id': f'weather_{col.split(" ")[0]}',
            'sales': df[col].values,  # reuse 'sales' column name to match the existing contract
            'day_of_week': df.index.dayofweek,
            'month': df.index.month,
            'day_of_year': df.index.dayofyear,
            'is_event': 0.0  # no events in weather data
        })
        all_data.append(series_df)
    
    result = pd.concat(all_data, ignore_index=True)
    return result
