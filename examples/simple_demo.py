"""
LiquidMind 简单演示

展示如何使用 LTC 和 CfC 进行时间序列预测
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from liquidmind import LTC, CfC, LiquidNetwork, LiquidForecaster


def generate_sine_wave(seq_len=100, num_samples=1, noise=0.1):
    """生成带噪声的正弦波数据"""
    t = np.linspace(0, 4 * np.pi, seq_len)
    data = []
    for _ in range(num_samples):
        wave = np.sin(t) + noise * np.random.randn(seq_len)
        data.append(wave)
    return np.array(data).reshape(num_samples, seq_len, 1)


def train_ltc_demo():
    """训练 LTC 模型示例"""
    print("=" * 50)
    print("LTC (Liquid Time-Constant) Demo")
    print("=" * 50)
    
    # 生成数据
    data = generate_sine_wave(seq_len=200, num_samples=100)
    
    # 划分训练集
    train_data = data[:80]
    test_data = data[80:]
    
    # 创建模型
    model = LiquidNetwork(
        input_size=1,
        hidden_size=32,
        output_size=1,
        mode="ltc",
        num_layers=1,
        dt=0.1
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 训练
    print("Training...")
    for epoch in range(50):
        model.train()
        total_loss = 0
        
        for i in range(len(train_data)):
            x = torch.tensor(train_data[i:i+1, :-1], dtype=torch.float32)
            y = torch.tensor(train_data[i:i+1, 1:], dtype=torch.float32)
            
            optimizer.zero_grad()
            pred, _ = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_data):.4f}")
    
    # 测试预测
    model.eval()
    with torch.no_grad():
        test_input = torch.tensor(test_data[0:1, :-10], dtype=torch.float32)
        predictions = model.predict(test_input, steps=10)
    
    print(f"\nPrediction shape: {predictions.shape}")
    print("LTC Demo completed!")
    
    return model


def train_cfc_demo():
    """训练 CfC 模型示例"""
    print("\n" + "=" * 50)
    print("CfC (Closed-form Continuous-time) Demo")
    print("=" * 50)
    
    # 生成数据
    data = generate_sine_wave(seq_len=200, num_samples=100)
    train_data = data[:80]
    
    # 创建模型
    model = LiquidNetwork(
        input_size=1,
        hidden_size=32,
        output_size=1,
        mode="cfc",
        num_layers=1
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 训练
    print("Training...")
    for epoch in range(50):
        model.train()
        total_loss = 0
        
        for i in range(len(train_data)):
            x = torch.tensor(train_data[i:i+1, :-1], dtype=torch.float32)
            y = torch.tensor(train_data[i:i+1, 1:], dtype=torch.float32)
            
            optimizer.zero_grad()
            pred, _ = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_data):.4f}")
    
    print("CfC Demo completed!")
    
    return model


def compare_models():
    """对比 LTC 和 CfC 的性能"""
    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)
    
    # 生成更复杂的数据
    t = np.linspace(0, 8 * np.pi, 500)
    data = np.sin(t) + 0.5 * np.sin(3 * t) + 0.1 * np.random.randn(500)
    data = data.reshape(1, 500, 1)
    
    # 划分
    train = data[:, :400, :]
    test = data[:, 400:, :]
    
    models = {
        "LTC": LiquidNetwork(1, 32, 1, mode="ltc", num_layers=1),
        "CfC": LiquidNetwork(1, 32, 1, mode="cfc", num_layers=1),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # 训练
        for epoch in range(30):
            model.train()
            x = torch.tensor(train[:, :-1, :], dtype=torch.float32)
            y = torch.tensor(train[:, 1:, :], dtype=torch.float32)
            
            optimizer.zero_grad()
            pred, _ = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        
        # 测试
        model.eval()
        with torch.no_grad():
            x_test = torch.tensor(train[:, -50:, :], dtype=torch.float32)
            pred = model.predict(x_test, steps=50)
            
            y_true = torch.tensor(test[:, :50, :], dtype=torch.float32)
            mse = criterion(pred, y_true).item()
        
        results[name] = mse
        print(f"{name} MSE: {mse:.6f}")
    
    print("\n" + "=" * 50)
    print("Summary:")
    for name, mse in results.items():
        print(f"  {name}: MSE = {mse:.6f}")


def stock_prediction_demo():
    """
    股票价格预测演示 (使用合成数据)
    
    实际应用时需要替换为真实股票数据
    """
    print("\n" + "=" * 50)
    print("Stock Price Prediction Demo")
    print("=" * 50)
    
    # 生成合成股价数据 (随机游走 + 趋势)
    np.random.seed(42)
    n_days = 252  # 一年交易日
    returns = np.random.randn(n_days) * 0.02
    price = 100 * np.exp(np.cumsum(returns))
    
    # 添加特征 (价格、成交量等)
    volume = np.random.randint(1000000, 10000000, n_days)
    
    # 归一化
    price_norm = (price - price.mean()) / price.std()
    volume_norm = (volume - volume.mean()) / volume.std()
    
    data = np.column_stack([price_norm, volume_norm])
    data = data.reshape(1, n_days, 2)
    
    # 划分训练/测试
    train_size = int(0.8 * n_days)
    train_data = data[:, :train_size, :]
    test_data = data[:, train_size:, :]
    
    # 创建预测模型
    model = LiquidForecaster(
        input_size=2,
        hidden_size=64,
        num_layers=2,
        forecast_horizon=5,
        mode="cfc"
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 训练
    print("Training on synthetic stock data...")
    for epoch in range(100):
        model.train()
        
        # 使用滑动窗口
        total_loss = 0
        for i in range(0, train_size - 20, 10):
            x = torch.tensor(train_data[:, i:i+15, :], dtype=torch.float32)
            y = torch.tensor(train_data[:, i+15:i+20, :], dtype=torch.float32)
            
            optimizer.zero_grad()
            pred, _ = model(x)
            loss = criterion(pred[:, -5:, :], y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    # 预测
    model.eval()
    with torch.no_grad():
        test_input = torch.tensor(test_data[:, :15, :], dtype=torch.float32)
        forecast = model.forecast(test_input, horizon=5)
    
    print(f"\nForecast shape: {forecast.shape}")
    print("Stock prediction demo completed!")
    
    return model


if __name__ == "__main__":
    print("LiquidMind - 液态神经网络演示")
    print("=" * 50)
    
    # 运行演示
    try:
        train_ltc_demo()
        train_cfc_demo()
        compare_models()
        stock_prediction_demo()
        
        print("\n" + "=" * 50)
        print("All demos completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
