# LiquidMind 🧠💧

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/olveww-dot/liquidmind?style=social)](https://github.com/olveww-dot/liquidmind)

> Liquid Neural Networks - Dynamic Adaptive Intelligence

Implementation of MIT's **Liquid Time-Constant (LTC)** and **Closed-form Continuous-time (CfC)** networks.

[简体中文](README.md) | English

## Core Concept

Traditional neural networks have fixed structures, but LiquidMind flows dynamically like water:

- **Dynamic Time Constants** - Neurons adjust response speed based on input
- **Continuous-time Modeling** - Uses differential equations instead of discrete time steps
- **Parameter Efficient** - Few neurons can express complex dynamics

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python examples/simple_demo.py
```

## Core Components

### 1. LTC (Liquid Time-Constant)
```python
from liquidmind import LTC

ltc = LTC(input_size=10, hidden_size=32, dt=0.1)
output, hidden = ltc(input, hidden)
```

### 2. CfC (Closed-form Continuous-time)
```python
from liquidmind import CfC

cfc = CfC(input_size=10, hidden_size=32)
output, hidden = cfc(input, hidden)
```

### 3. LiquidNetwork (Complete Network)
```python
from liquidmind import LiquidNetwork

model = LiquidNetwork(
    input_size=1,
    hidden_size=64,
    output_size=1,
    mode="cfc",  # or "ltc"
    num_layers=2
)
```

### 4. LiquidForecaster (Time Series Forecasting)
```python
from liquidmind import LiquidForecaster

forecaster = LiquidForecaster(
    input_size=2,  # Number of features
    hidden_size=64,
    forecast_horizon=5,  # Predict 5 steps ahead
    mode="cfc"
)

# Predict after training
forecast = forecaster.forecast(history_data)
```

## Use Cases

- 📈 **Stock Price Prediction** - Capture market dynamics
- 🔬 **Sensor Data Analysis** - Irregular time series
- 🚗 **Autonomous Driving** - Real-time continuous control
- 🏥 **Medical Signal Processing** - ECG/EEG analysis
- 📊 **Edge Device Deployment** - Few parameters, fast inference

## Architecture Comparison

| Feature | LTC | CfC |
|---------|-----|-----|
| Computation | Euler Integration | Closed-form Solution |
| Speed | Slower | Faster |
| Accuracy | High | High |
| Stability | Requires tuning | More stable |
| Recommended Use | Research/Fine modeling | Production/Real-time |

## Project Structure

```
liquidmind/
├── liquidmind/
│   ├── __init__.py
│   ├── ltc.py          # LTC implementation
│   ├── cfc.py          # CfC implementation
│   └── liquid_layer.py # Unified interface
├── examples/
│   └── simple_demo.py  # Demo code
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## References

1. Hasani et al. "Liquid Time-Constant Networks" (2021)
2. Hasani et al. "Closed-form Continuous-time Neural Networks" (2022)
3. [ncps](https://github.com/mlech26l/ncps) - Official PyTorch implementation reference

## License

MIT License

---
*Built with ❤️ by EC*
