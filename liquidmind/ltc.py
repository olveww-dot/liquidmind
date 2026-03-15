"""
Liquid Time-Constant (LTC) 层实现

基于论文: "Liquid Time-Constant Networks" (Hasani et al., 2021)
核心思想: 使用连续时间动力系统替代离散RNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LTC(nn.Module):
    """
    Liquid Time-Constant 网络层
    
    特点:
    - 神经元具有可变的时间常数 tau
    - 时间常数由输入动态决定
    - 使用欧拉方法求解微分方程
    
    微分方程: dx/dt = -x/tau + f(x, I)
    其中 tau 是输入依赖的时间常数
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dt: float = 0.1,
        tau_min: float = 0.1,
        tau_max: float = 10.0,
        activation: str = "tanh"
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        
        # 输入到隐层的权重
        self.input_weights = nn.Linear(input_size, hidden_size)
        
        # 隐层到隐层的权重 (循环连接)
        self.recurrent_weights = nn.Linear(hidden_size, hidden_size)
        
        # 时间常数参数 - 输入依赖
        self.tau_weights = nn.Linear(input_size + hidden_size, hidden_size)
        
        # 激活函数
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        else:
            self.activation = torch.tanh
    
    def _compute_tau(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        计算输入依赖的时间常数 tau
        
        tau 决定了系统对输入变化的响应速度:
        - 小 tau: 快速响应，但可能不稳定
        - 大 tau: 慢速响应，更平滑
        """
        combined = torch.cat([x, h], dim=-1)
        # 使用 sigmoid 将 tau 限制在 [tau_min, tau_max] 范围
        tau = torch.sigmoid(self.tau_weights(combined))
        tau = self.tau_min + tau * (self.tau_max - self.tau_min)
        return tau
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_size]
            hidden: 上一时刻隐状态 [batch_size, hidden_size]
        
        Returns:
            output: 输出 [batch_size, hidden_size]
            new_hidden: 新隐状态 [batch_size, hidden_size]
        """
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # 计算时间常数 tau
        tau = self._compute_tau(x, hidden)
        
        # 计算输入和循环信号
        input_signal = self.input_weights(x)
        recurrent_signal = self.recurrent_weights(hidden)
        
        # 计算神经元的总输入
        total_input = input_signal + recurrent_signal
        
        # 应用激活函数
        f_h = self.activation(total_input)
        
        # 欧拉方法求解微分方程: dh/dt = (-h + f_h) / tau
        # 离散化: h_new = h + dt * (-h + f_h) / tau
        decay = self.dt / tau
        new_hidden = hidden + decay * (f_h - hidden)
        
        return new_hidden, new_hidden


class LTCSequence(nn.Module):
    """
    序列版本的 LTC，处理时间序列数据
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dt: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 堆叠多层 LTC
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input = input_size if i == 0 else hidden_size
            self.layers.append(LTC(layer_input, hidden_size, dt))
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(
        self, 
        x: torch.Tensor,
        hidden: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: 输入序列 [batch_size, seq_len, input_size]
        Returns:
            output: 输出序列 [batch_size, seq_len, output_size]
        """
        batch_size, seq_len, _ = x.size()
        
        if hidden is None:
            hidden = [None] * self.num_layers
        
        outputs = []
        current_hidden = hidden.copy()
        
        for t in range(seq_len):
            layer_input = x[:, t, :]
            
            # 逐层传播
            for i, layer in enumerate(self.layers):
                layer_output, current_hidden[i] = layer(layer_input, current_hidden[i])
                layer_input = layer_output
            
            # 计算输出
            output = self.output_layer(layer_input)
            outputs.append(output)
        
        # 堆叠输出
        outputs = torch.stack(outputs, dim=1)
        
        return outputs, current_hidden
