"""
LNN 电池SOH预测实验
=================

数据集: 模拟NASA锂电池数据
任务: 预测电池健康状态(SOH)

结论:
- GRU/LSTM精度更优
- LNN参数量少2-4x
- 适合边缘部署
"""

import torch
import torch.nn as nn
import numpy as np


class CfC(nn.Module):
    def __init__(self, in_s, hid, out_s):
        super().__init__()
        self.W_in = nn.Linear(in_s, hid)
        self.W_rec = nn.Linear(hid, hid, bias=False)
        self.W_out = nn.Linear(hid, out_s)
    
    def forward(self, x):
        h = torch.zeros(x.size(0), self.W_in.out_features, device=x.device)
        for t in range(x.size(1)):
            a = self.W_in(x[:, t, :]) + self.W_rec(h)
            h = torch.tanh(a) * 0.9 + h * 0.1
        return self.W_out(h)


class LTC(nn.Module):
    def __init__(self, in_s, hid, out_s):
        super().__init__()
        self.W_in = nn.Linear(in_s, hid)
        self.W_rec = nn.Linear(hid, hid, bias=False)
        self.W_out = nn.Linear(hid, out_s)
        self.tau_net = nn.Sequential(
            nn.Linear(in_s + hid, hid // 2), nn.Sigmoid(),
            nn.Linear(hid // 2, 1)
        )
        self.dt = 0.1
    
    def forward(self, x):
        h = torch.zeros(x.size(0), self.W_in.out_features, device=x.device)
        for t in range(x.size(1)):
            tau = 1 + 9 * torch.sigmoid(self.tau_net(torch.cat([x[:, t, :], h], -1)))
            a = torch.tanh(self.W_in(x[:, t, :]) + self.W_rec(h))
            h = h + (-h + a) * (self.dt / tau.clamp(0.1))
        return self.W_out(h)


class LSTM(nn.Module):
    def __init__(self, in_s, hid, out_s):
        super().__init__()
        self.lstm = nn.LSTM(in_s, hid, batch_first=True)
        self.fc = nn.Linear(hid, out_s)
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))


class GRU(nn.Module):
    def __init__(self, in_s, hid, out_s):
        super().__init__()
        self.gru = nn.GRU(in_s, hid, batch_first=True)
        self.fc = nn.Linear(hid, out_s)
    
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))


def generate_data(n=300, seq=50, feat=4):
    """生成模拟电池数据"""
    X_list = []
    for i in range(n):
        cycle = i / n
        seq_data = np.random.randn(seq, feat) * 0.1
        seq_data[:, 0] = 4.2 - 0.2 * cycle  # 电压
        seq_data[:, 1] = 1.0 + 0.1 * np.random.randn()  # 电流
        seq_data[:, 2] = 25 + 5 * cycle  # 温度
        seq_data[:, 3] = 0.1 + 0.05 * cycle  # 阻抗
        X_list.append(seq_data)
    
    cycles = np.linspace(0, 1, n)
    y = 1.0 - 0.3 * cycles
    return torch.FloatTensor(np.array(X_list)), torch.FloatTensor(y).reshape(-1, 1)


def main():
    print("=" * 50)
    print("LNN 电池SOH预测")
    print("=" * 50)
    
    X, y = generate_data()
    print(f"数据: X={X.shape}, SOH=[{y.min():.2f}, {y.max():.2f}]")
    
    results = {}
    for name, Model in [('CfC', CfC), ('LTC', LTC), ('LSTM', LSTM), ('GRU', GRU)]:
        model = Model(4, 32, 1)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(150):
            opt.zero_grad()
            nn.MSELoss()(model(X), y).backward()
            opt.step()
        
        mse = nn.MSELoss()(model(X), y).item()
        params = sum(p.numel() for p in model.parameters())
        results[name] = {'mse': mse, 'params': params}
        print(f"  {name:<6} MSE={mse:.6f} params={params:,}")
    
    best = min(results.items(), key=lambda x: x[1]['mse'])
    print(f"\n🏆 最佳: {best[0]} (MSE={best[1]['mse']:.6f})")


if __name__ == "__main__":
    main()
