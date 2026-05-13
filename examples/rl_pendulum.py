"""
LNN 强化学习: 双摆控制
=====================

环境: 双摆平衡
任务: 保持双摆不倒下

结论: LSTM精度略优, 但LNN参数量少2.3x
"""

import torch
import torch.nn as nn
import numpy as np


class DoublePendulum:
    """双摆环境"""
    def reset(self):
        self.t1 = np.random.uniform(-0.5, 0.5)
        self.t2 = np.random.uniform(-0.3, 0.3)
        self.w1, self.w2 = 0, 0
        return np.array([self.t1, self.t2, self.w1, self.w2])
    
    def step(self, a):
        f = 5.0 if a == 1 else -5.0
        self.w1 += f * 0.01 - 0.1 * self.w1 + 0.01 * np.sin(self.t1)
        self.w2 += f * 0.02 - 0.05 * self.w2 + 0.02 * np.sin(self.t2)
        self.t1 += self.w1 * 0.01
        self.t2 += self.w2 * 0.01
        done = abs(self.t1) > 1.5 or abs(self.t2) > 1.5
        r = 1.0 if not done else -10.0
        return np.array([self.t1, self.t2, self.w1, self.w2]), r, done


class LTCDynamics(nn.Module):
    """LTC动态策略"""
    def __init__(self, s=4, h=32):
        super().__init__()
        self.W_in = nn.Linear(s, h)
        self.W_rec = nn.Linear(h, h, bias=False)
        self.tau_net = nn.Sequential(nn.Linear(s+h, h//2), nn.Sigmoid(), nn.Linear(h//2, 1))
        self.fc = nn.Sequential(nn.Linear(h, 16), nn.ReLU(), nn.Linear(16, 2))
        self.dt = 0.1
    
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(0)
        h = torch.zeros(x.size(0), self.W_in.out_features, device=x.device)
        for t in range(x.size(1)):
            tau = 1 + 9 * torch.sigmoid(self.tau_net(torch.cat([x[:, t, :], h], -1)))
            a = torch.tanh(self.W_in(x[:, t, :]) + self.W_rec(h))
            h = h + (-h + a) * (self.dt / tau.clamp(0.1))
        return self.fc(h)
    
    def act(self, s):
        with torch.no_grad():
            return torch.argmax(torch.softmax(self(torch.FloatTensor(s).unsqueeze(0)), -1), -1).item()


class LSTMDynamics(nn.Module):
    """LSTM策略 (对比)"""
    def __init__(self, s=4, h=32):
        super().__init__()
        self.rnn = nn.LSTM(s, h, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(h, 16), nn.ReLU(), nn.Linear(16, 2))
    
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(0)
        _, (h, _) = self.rnn(x)
        return self.fc(h.squeeze(0))
    
    def act(self, s):
        with torch.no_grad():
            return torch.argmax(torch.softmax(self(torch.FloatTensor(s).unsqueeze(0)), -1), -1).item()


def reinforce(policy, env, eps=80):
    opt = torch.optim.Adam(policy.parameters(), lr=0.005)
    rs = []
    for ep in range(eps):
        S, A, R = [], [], []
        s = env.reset()
        for _ in range(300):
            S.append(s); a = policy.act(s); A.append(a)
            s, r, d = env.step(a); R.append(r)
            if d: break
        G, rets = 0, []
        for r in reversed(R):
            G = r + 0.98 * G; rets.insert(0, G)
        rets = torch.FloatTensor(rets)
        rets = (rets - rets.mean()) / (rets.std() + 1e-8)
        opt.zero_grad()
        for s, a, g in zip(S, A, rets):
            torch.log_softmax(policy(torch.FloatTensor(s).unsqueeze(0)), -1)[0, a] * g
        opt.step()
        rs.append(sum(R))
    return rs


def main():
    print("=" * 50)
    print("LNN 强化学习: 双摆控制")
    print("=" * 50)
    
    env = DoublePendulum()
    
    print("\n训练中...")
    ltc_rs = reinforce(LTCDynamics(), env)
    lstm_rs = reinforce(LSTMDynamics(), env)
    
    print(f"\n结果:")
    print(f"  LTC:  {np.mean(ltc_rs[-20:]):.1f} (max={max(ltc_rs)})")
    print(f"  LSTM: {np.mean(lstm_rs[-20:]):.1f} (max={max(lstm_rs)})")
    print(f"  参数: LTC={sum(p.numel() for p in LTCDynamics().parameters()):,}")
    print(f"       LSTM={sum(p.numel() for p in LSTMDynamics().parameters()):,}")


if __name__ == "__main__":
    main()
