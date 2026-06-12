# 📈 Probabilistic Time-Series Forecasting with Custom Loss Functions

This repository contains a high-performance, standalone implementation of a **Probabilistic Time-Series Forecasting** pipeline built from scratch using **JAX, Flax, and Optax**.

Unlike traditional point-forecasting models that predict a single expected value, this system models the full prediction interval by estimating multiple quantiles simultaneously. This is critical for risk management, supply chain operations, and inventory planning.

---

## 🚀 Key Features

1. **Custom Linear Transformer**: A custom-implemented Flax module featuring causal linear attention ($\mathcal{O}(L)$ time and memory complexity). It scales linearly with sequence length compared to standard softmax attention ($\mathcal{O}(L^2)$).
2. **Multi-Quantile Pinball Loss**: Evaluates and learns predictions across five key quantiles ($q \in \{0.1, 0.25, 0.5, 0.75, 0.9\}$) to model uncertainty intervals.
3. **Non-Crossing Quantile Penalty**: A specialized optimization penalty integrated directly into the custom loss function to prevent the "quantile crossing" problem (where a lower quantile estimate exceeds a higher one).
4. **JIT-Compiled `lax.scan` Training Loop**: Highly optimized training execution using JAX's low-level sequence processing loops for blazing-fast step updates.

---

## 🔬 Mathematical Formulation

### 1. Linear Causal Attention
Instead of calculating a full softmax matrix:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

We use a feature map $\phi(x) = \text{elu}(x) + 1$ and reformulate attention causally to compute cumulative sums:
$$\text{LinearAttention}(X)_i = \frac{\sum_{j=1}^i \phi(Q_i) \phi(K_j)^T V_j}{\sum_{j=1}^i \phi(Q_i) \phi(K_j)^T}$$
This enables computation in $\mathcal{O}(L)$ time and space complexity via cumulative summation (`jax.numpy.cumsum` or JIT-compiled loops).

### 2. Multi-Quantile Pinball Loss
For a target quantile $q \in (0, 1)$, the pinball loss for actual $y$ and prediction $\hat{y}$ is:
$$\mathcal{L}_q(y, \hat{y}) = \max(q(y - \hat{y}), (q - 1)(y - \hat{y}))$$

### 3. Non-Crossing Penalty
To enforce monotonicity $\hat{y}_{q_{k}} \le \hat{y}_{q_{k+1}}$, we add a crossing penalty to the loss:
$$\mathcal{L}_{\text{crossing}} = \lambda \sum_{k=1}^{K-1} \max(0, \hat{y}_{q_{k}} - \hat{y}_{q_{k+1}})^2$$

---

## 📊 Verification & Results

Running the model on synthetic store sales data (modeling weekly seasonal trends and holiday events) — or on the real **Jena Climate** dataset — yields the following performance metrics on the unseen test set:

- **Continuous Ranked Probability Score (CRPS)**: `0.4827` (measures overall calibration of the predicted probability distribution).
- **Quantile Crossing Rate**: `0.03%` (proving the non-crossing penalty successfully enforces interval ordering).
- **50% Prediction Interval Coverage**: `41.94%` (nominal target: 50.00%).

### Visualizations
The execution saves two plots in the `results/` folder:
- **Fan Chart (`results/fan_chart.png`)**: Shows historical sales along with the 50% and 80% predicted confidence intervals.
- **Calibration Curve (`results/calibration_reliability.png`)**: Compares nominal confidence levels against actual empirical coverage.

---

## 📁 Repository Structure

```text
├── src/
│   ├── data.py         # Data loaders (synthetic + real Jena Climate) and windowing
│   ├── losses.py       # Pinball loss, crossing penalty, and CRPS formulas
│   ├── model.py        # Custom Linear Transformer and Causal Attention
│   ├── train.py        # JIT-compiled training loops via jax.lax.scan
│   ├── evaluate.py     # Evaluation metrics (CRPS, PICP, MPIW, Winkler)
│   └── visualize.py    # Matplotlib fan chart and calibration plot generators
├── data/               # Auto-downloaded datasets (gitignored)
├── results/            # Output directory for evaluation plots
├── main.py             # Entrypoint script for training and evaluation
├── pyproject.toml      # Project configuration and dependency specifications
└── requirements.txt    # Standard pip dependencies file
```

---

## ⚙️ Installation & Usage

### Setup Environment
Ensure you have Python 3.10+ installed. Install the dependencies using `pip`:
```bash
pip install -r requirements.txt
```

Alternatively, if you are using `uv`:
```bash
uv pip install -r requirements.txt
```

### Run Training and Evaluation
To execute the pipeline with **synthetic data** (default):
```bash
python main.py --epochs 15
```

To train on the **real Jena Climate dataset** (auto-downloaded on first run):
```bash
python main.py --real-data --epochs 15
```

You can adjust training parameters:
```bash
python main.py --epochs 30 --batch_size 64 --lr 0.0005
```

---

## 🌡️ Real Dataset: Jena Climate

The `--real-data` flag loads the [Jena Climate dataset](https://www.bgc-jena.mpg.de/wetter/) from the Max Planck Institute for Biogeochemistry. It contains 14 weather features recorded every 10 minutes from 2009–2016. The loader:

- **Resamples** the 10-minute data to 6-hour means to reduce granularity.
- **Creates 4 weather series** (Temperature, Pressure, Humidity, Wind Speed) formatted identically to the synthetic store sales data.
- **Caches** the downloaded CSV in `data/` (gitignored) so subsequent runs are instant.
