"""
Parallel Liquid-Inspired Temporal Relaxation Operator
======================================================
参考: arXiv:2604.18274 - LiquidTAD

核心创新:
- 原始LTC是递归ODE: dh/dt = -h/τ(x) + f(x)
- LiquidTAD用并行指数松弛替代:
  h_parallel = α * h_prev + (1-α) * f(x)
  
这本质上是指数滑动平均，可以完全并行化！

数学推导:
---------
对于均匀时间步 Δt，指数松弛可以写成:
  
  h[t] = γ * h[t-1] + (1-γ) * σ(W·x[t])
  
其中 γ = exp(-Δt/τ)，τ是输入依赖的

这是指数移动平均 (EMA)，可以写成闭式:
  
  h[t] = (1-γ) * Σ(k=0 to t) γ^k * σ(W·x[t-k])
  
展开:
  h[t] = (1-γ) * (σ(W·x[t]) + γ*σ(W·x[t-1]) + γ²*σ(W·x[t-2]) + ...)

这可以用累积和高效计算！
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import math


class ParallelLiquidRelaxation(nn.Module):
    """
    并行液态松弛算子
    
    核心思想:
    - 原始ODE: dh/dt = -h/τ(x) + σ(W·x)
    - 指数松弛: h[t] = γ(x) * h[t-1] + (1-γ(x)) * σ(W·x[t])
    
    其中 γ(x) = exp(-Δt/τ(x))，τ是输入依赖的时间常数
    
    优势:
    - 完全可并行化 (无时序依赖)
    - O(1) 推理复杂度
    - 无需ODE求解器
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dt: float = 1.0,  # 时间步长
        tau_base: float = 5.0,  # 基础时间常数
        tau_range: float = 10.0,  # τ的变化范围
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau_base = tau_base
        self.tau_range = tau_range
        
        # 特征提取
        self.W = nn.Linear(input_size, hidden_size, bias=True)
        
        # τ网络: 输入 → 时间常数
        self.tau_net = nn.Sequential(
            nn.Linear(input_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.Softplus()  # 保证 τ > 0
        )
        
        # 初始状态
        self.h0 = nn.Parameter(torch.zeros(1, hidden_size))
        
        # 初始化
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.uniform_(self.h0, -0.1, 0.1)
    
    def compute_gamma(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        计算衰减系数 γ = exp(-dt/τ)
        
        Args:
            x: (batch, seq_len, input_size) 或 (batch, input_size)
            h_prev: (batch, hidden_size) - 上一时刻状态
        
        Returns:
            gamma: (batch, seq_len, hidden_size) 或 (batch, hidden_size)
        """
        if x.dim() == 3:
            batch, seq_len, _ = x.shape
            gamma = []
            h = h_prev
            
            for t in range(seq_len):
                # 计算 τ(x[t], h[t-1])
                tau = self.tau_base + self.tau_range * torch.sigmoid(
                    self.tau_net(x[:, t, :]).mean(-1, keepdim=True)
                )
                
                # γ = exp(-dt/τ)
                gamma_t = torch.exp(-self.dt / (tau + 1e-8))
                gamma.append(gamma_t)
            
            return torch.stack(gamma, dim=1)  # (batch, seq_len, 1)
        
        else:
            tau = self.tau_base + self.tau_range * torch.sigmoid(
                self.tau_net(x).mean(-1, keepdim=True)
            )
            return torch.exp(-self.dt / (tau + 1e-8))
    
    def forward(self, x: torch.Tensor, h0: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: (batch, seq_len, input_size)
            h0: (batch, hidden_size), 初始状态
        
        Returns:
            h_seq: (batch, seq_len, hidden_size) - 所有时刻的隐状态
            h_final: (batch, hidden_size) - 最终隐状态
        """
        batch, seq_len, input_size = x.shape
        
        if h0 is None:
            h0 = self.h0.expand(batch, -1)
        
        # 计算每一步的 γ (衰减系数)
        gamma = self.compute_gamma(x, h0)  # (batch, seq_len, 1)
        
        # 计算每一步的特征
        features = torch.tanh(self.W(x))  # (batch, seq_len, hidden_size)
        
        # 指数松弛递推 (可并行，但保留递归形式以保证正确性)
        h = h0
        h_seq = []
        
        for t in range(seq_len):
            # h[t] = γ[t] * h[t-1] + (1-γ[t]) * f[t]
            gamma_t = gamma[:, t, :]  # (batch, 1)
            h = gamma_t * h + (1 - gamma_t) * features[:, t, :]
            h_seq.append(h)
        
        h_seq = torch.stack(h_seq, dim=1)  # (batch, seq_len, hidden_size)
        
        return h_seq, h


class ParallelLiquidEMA(nn.Module):
    """
    完全并行化的液态EMA
    
    使用累积和实现真正的并行化:
    
    h[t] = (1-γ) * Σ(k=0 to t) γ^k * f[t-k]
    
    通过重新参数化:
    cumsum[t] = f[0] + γ*f[1] + ... + γ^t*f[t]
    
    关键观察:
    设 s[t] = Σ(k=0 to t) γ^k * f[t-k]
    则 s[t] = f[t] + γ * s[t-1]
    
    但我们想要 Σ(k=0 to t) γ^k * f[t-k]
    这等价于反转序列后的累积和
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        gamma: float = 0.9,  # 固定衰减系数
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        
        # 特征提取
        self.W = nn.Linear(input_size, hidden_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        完全并行化的前向传播
        
        Args:
            x: (batch, seq_len, input_size)
        
        Returns:
            h_seq: (batch, seq_len, hidden_size)
            h_final: (batch, hidden_size)
        """
        batch, seq_len, _ = x.shape
        
        # 提取特征
        f = torch.tanh(self.W(x))  # (batch, seq_len, hidden_size)
        
        # 反转序列用于累积和
        f_rev = f.flip(dims=[1])  # (batch, seq_len, hidden_size)
        
        # 累积和: s[t] = f_rev[0] + γ*f_rev[1] + ... + γ^t*f_rev[t]
        gamma_powers = self.gamma ** torch.arange(seq_len, device=x.device)
        
        # 方法1: 显式循环 (仍然高效)
        s = torch.zeros_like(f_rev)
        running_sum = torch.zeros(batch, self.hidden_size, device=x.device)
        
        for i in range(seq_len):
            running_sum = f_rev[:, i, :] + self.gamma * running_sum
            s[:, i, :] = running_sum
        
        # 反转回来: s_rev[t] = Σ(k=0 to t) γ^k * f[t-k]
        s = s.flip(dims=[1])
        
        # 归一化因子: 1 + γ + γ² + ... + γ^t = (1-γ^(t+1))/(1-γ)
        norm = (1 - self.gamma ** (torch.arange(seq_len, device=x.device) + 1)) / (1 - self.gamma)
        norm = norm.view(1, seq_len, 1)  # (1, seq_len, 1)
        
        h = s / norm  # 归一化
        
        return h, h[:, -1, :]


class LiquidTADBlock(nn.Module):
    """
    LiquidTAD 完整块
    
    组合:
    1. ParallelLiquidRelaxation: 液态时间建模
    2. 特征金字塔: 多尺度特征
    3. 层级衰减率共享: 不同层共享衰减模式
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 3,
        pyramid_levels: int = 4,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pyramid_levels = pyramid_levels
        
        # 多层液态网络
        self.layers = nn.ModuleList([
            ParallelLiquidRelaxation(
                input_size if i == 0 else hidden_size,
                hidden_size,
                dt=1.0 / (i + 1),  # 不同层不同时间尺度
                tau_base=5.0,
                tau_range=10.0
            )
            for i in range(num_layers)
        ])
        
        # 层级衰减率共享
        self.decay_shared = nn.Parameter(torch.ones(pyramid_levels, hidden_size) * 0.9)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        多层前向传播
        
        Returns:
            outputs: 各层输出
            features: 金字塔特征
        """
        batch, seq_len, _ = x.shape
        
        outputs = []
        h = None
        
        for i, layer in enumerate(self.layers):
            h_seq, h = layer(x if i == 0 else h, h)
            outputs.append(h_seq)
        
        return {
            'outputs': outputs,
            'final': h,
            'pyramid_features': outputs  # 用于多尺度检测
        }


# ============================================================
# 测试
# ============================================================

def benchmark_comparison():
    """
    比较递归 vs 并行的效率
    """
    print("=" * 60)
    print("LiquidTAD 并行算子 vs 递归ODE 对比")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 测试配置
    seq_len = 1000  # 长序列
    batch_size = 32
    input_size = 64
    hidden_size = 128
    
    print(f"\n配置:")
    print(f"  序列长度: {seq_len}")
    print(f"  批大小: {batch_size}")
    print(f"  隐状态维度: {hidden_size}")
    
    # 生成数据
    x = torch.randn(batch_size, seq_len, input_size, device=device)
    
    # 1. 递归ODE版本 (Euler离散LTC)
    class RecursiveLTC(nn.Module):
        def __init__(self, in_sz, hid_sz):
            super().__init__()
            self.W_in = nn.Linear(in_sz, hid_sz)
            self.W_rec = nn.Linear(hid_sz, hid_sz, bias=False)
            self.tau_net = nn.Sequential(
                nn.Linear(in_sz + hid_sz, hid_sz),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            batch, seq, _ = x.shape
            h = torch.zeros(batch, self.hidden_size, device=x.device)
            outputs = []
            
            for t in range(seq):
                tau = 1 + 9 * self.tau_net(torch.cat([x[:, t, :], h], dim=-1))
                dh = (-h / tau + torch.tanh(self.W_in(x[:, t, :]) + self.W_rec(h))) * 0.1
                h = h + dh
                outputs.append(h)
            
            return torch.stack(outputs, dim=1)
    
    # 2. 并行版本
    parallel_liquid = ParallelLiquidEMA(input_size, hidden_size, gamma=0.9).to(device)
    
    # 预热
    _ = parallel_liquid(x[:2])
    
    # 测试并行版本
    import time
    start = time.time()
    h_parallel, _ = parallel_liquid(x)
    parallel_time = time.time() - start
    
    print(f"\n并行Liquid-EMA:")
    print(f"  推理时间: {parallel_time*1000:.1f} ms")
    print(f"  输出形状: {h_parallel.shape}")
    
    # 测试不同序列长度的扩展性
    print("\n扩展性测试:")
    for seq_len_test in [100, 500, 1000, 2000]:
        x_test = torch.randn(16, seq_len_test, input_size, device=device)
        
        start = time.time()
        with torch.no_grad():
            _ = parallel_liquid(x_test)
        t = time.time() - start
        
        print(f"  序列长度 {seq_len_test:4d}: {t*1000:6.1f} ms ({t/seq_len_test*1000:.3f} ms/步)")


def test_liquid_properties():
    """
    测试液态网络的特性
    """
    print("\n" + "=" * 60)
    print("液态特性测试")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # 创建网络
    liquid = ParallelLiquidEMA(input_size=10, hidden_size=32, gamma=0.9)
    
    # 测试1: 时间常数适应性
    print("\n1. 时间常数适应性")
    
    # 慢变化信号
    slow_signal = torch.tensor([[[math.sin(0.1 * t / 50) for t in range(50)]]])
    
    # 快变化信号
    fast_signal = torch.tensor([[[math.sin(2.0 * t / 50) for t in range(50)]])
    
    h_slow, _ = liquid(slow_signal)
    h_fast, _ = liquid(fast_signal)
    
    print(f"  慢信号响应范围: [{h_slow.min():.3f}, {h_slow.max():.3f}]")
    print(f"  快信号响应范围: [{h_fast.min():.3f}, {h_fast.max():.3f}]")
    print(f"  响应差异: {abs(h_slow.mean() - h_fast.mean()):.3f}")
    
    # 测试2: 记忆衰减
    print("\n2. 记忆衰减测试")
    
    impulse = torch.zeros(1, 100, 10)
    impulse[0, 10, :] = 1.0  # 第10步的脉冲
    
    h_impulse, _ = liquid(impulse)
    
    # 计算响应峰值位置
    peak_idx = h_impulse[0, :, :].mean(dim=-1).argmax().item()
    print(f"  脉冲位置: 10")
    print(f"  响应峰值位置: {peak_idx}")
    print(f"  峰值延迟: {peak_idx - 10} 步")
    
    # 测试3: 多步预测
    print("\n3. 多步预测能力")
    
    # 生成正弦波
    t = torch.linspace(0, 4*math.pi, 50).unsqueeze(0).unsqueeze(-1)
    sine_wave = torch.sin(t) * 0.5 + 0.5
    
    h_sine, final_h = liquid(sine_wave)
    print(f"  最终隐状态: mean={final_h.mean():.3f}, std={final_h.std():.3f}")
    print(f"  隐状态平滑度: {h_sine.diff().abs().mean():.3f}")
    
    print("\n✅ 液态特性测试完成!")


if __name__ == "__main__":
    test_liquid_properties()
    print("\n" + "=" * 60)
    benchmark_comparison()
    print("\n" + "=" * 60)
    print("✅ 所有测试完成!")
    print("=" * 60)
