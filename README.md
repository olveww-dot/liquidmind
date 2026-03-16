# LiquidMind 🧠💧

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/olveww-dot/liquidmind?style=social)](https://github.com/olveww-dot/liquidmind)

> 液态神经网络 - 动态适应的智能

基于 MIT 的 **Liquid Time-Constant (LTC)** 和 **Closed-form Continuous-time (CfC)** 网络实现。

<!-- [English](README_EN.md) | -->简体中文

## 核心概念

传统神经网络结构固定，LiquidMind 像水一样**动态流动**:

- **时间常数动态变化** - 神经元根据输入调整响应速度
- **连续时间建模** - 用微分方程替代离散时间步
- **参数高效** - 少量神经元即可表达复杂动态

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行演示
python examples/simple_demo.py
```

## 核心组件

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

### 3. LiquidNetwork (完整网络)
```python
from liquidmind import LiquidNetwork

model = LiquidNetwork(
    input_size=1,
    hidden_size=64,
    output_size=1,
    mode="cfc",  # 或 "ltc"
    num_layers=2
)
```

### 4. LiquidForecaster (时间序列预测)
```python
from liquidmind import LiquidForecaster

forecaster = LiquidForecaster(
    input_size=2,  # 特征数
    hidden_size=64,
    forecast_horizon=5,  # 预测5步
    mode="cfc"
)

# 训练后预测
forecast = forecaster.forecast(history_data)
```

## 应用场景

- 📈 **股票价格预测** - 捕捉市场动态
- 🔬 **传感器数据分析** - 不规则时间序列
- 🚗 **自动驾驶决策** - 实时连续控制
- 🏥 **医疗信号处理** - ECG/EEG 分析
- 📊 **边缘设备部署** - 参数少，推理快

## 架构对比

| 特性 | LTC | CfC |
|------|-----|-----|
| 计算方式 | 欧拉积分 | 闭式解 |
| 速度 | 较慢 | 更快 |
| 精度 | 高 | 高 |
| 稳定性 | 需调参 | 更稳定 |
| 推荐场景 | 研究/精细建模 | 生产/实时应用 |

## 项目结构

```
liquidmind/
├── liquidmind/
│   ├── __init__.py
│   ├── ltc.py          # LTC 实现
│   ├── cfc.py          # CfC 实现
│   └── liquid_layer.py # 通用接口
├── examples/
│   └── simple_demo.py  # 演示代码
├── requirements.txt
└── README.md
```

## 参考文献

1. Hasani et al. "Liquid Time-Constant Networks" (2021)
2. Hasani et al. "Closed-form Continuous-time Neural Networks" (2022)
3. [ncps](https://github.com/mlech26l/ncps) - 官方 PyTorch 实现参考

## 许可证

MIT License

---
*Built with ❤️ by EC*
