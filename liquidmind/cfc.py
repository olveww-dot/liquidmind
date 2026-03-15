"""
Closed-form Continuous-time (CfC) 层实现

基于论文: "Closed-form Continuous-time Neural Networks" (Hasani et al., 2022)
核心思想: 使用闭式解替代数值积分，大幅提高计算效率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class CfC(nn.Module):
    """
    Closed-form Continuous-time 网络层
    
    相比 LTC 的优势:
    - 闭式解，无需欧拉积分
    - 计算效率更高
    - 训练更稳定
    
    核心方程: h(t) = h0 * exp(-t/tau) + (1 - exp(-t/tau)) * f(x)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mode: str = "default",
        activation: str = "tanh",
        backbone_layers: int = 1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        
        # 输入投影
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # 隐状态投影
        self.hidden_projection = nn.Linear(hidden_size, hidden_size)
        
        # 时间常数网络
        self.tau_net = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Softplus()  # 确保 tau > 0
        )
        
        # 选择信号网络 (用于插值)
        self.interpolation_net = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # 激活函数
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            self.activation = torch.tanh
    
    def _compute_parameters(
        self, 
        x: torch.Tensor, 
        h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 CfC 的关键参数
        
        Returns:
            tau: 时间常数
            interpolation: 插值系数 (0-1)
            f_x: 输入变换后的信号
        """
        combined = torch.cat([x, h], dim=-1)
        
        # 时间常数 (决定记忆长度)
        tau = self.tau_net(combined)
        tau = torch.clamp(tau, min=0.01, max=100.0)
        
        # 插值系数 (平衡新旧信息)
        interpolation = self.interpolation_net(combined)
        
        # 输入信号
        f_x = self.activation(self.input_projection(x))
        
        return tau, interpolation, f_x
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        timespan: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入 [batch_size, input_size]
            hidden: 上一时刻隐状态 [batch_size, hidden_size]
            timespan: 时间跨度 (用于连续时间建模)
        
        Returns:
            output: 输出
            new_hidden: 新隐状态
        """
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # 计算参数
        tau, interpolation, f_x = self._compute_parameters(x, hidden)
        
        # 隐状态变换
        h_transformed = self.hidden_projection(hidden)
        
        # CfC 闭式解
        # h_new = interpolation * f_x + (1 - interpolation) * h * exp(-timespan/tau)
        decay = torch.exp(-timespan / tau)
        new_hidden = interpolation * f_x + (1 - interpolation) * h_transformed * decay
        
        return new_hidden, new_hidden


class CfCSequence(nn.Module):
    """
    序列版本的 CfC
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        mode: str = "default"
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 堆叠多层 CfC
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input = input_size if i == 0 else hidden_size
            self.layers.append(CfC(layer_input, hidden_size, mode))
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[list] = None,
        timespan: float = 1.0
    ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: 输入序列 [batch_size, seq_len, input_size]
            timespan: 每个时间步的时间跨度
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
                layer_output, current_hidden[i] = layer(
                    layer_input, current_hidden[i], timespan
                )
                layer_input = layer_output
            
            # 计算输出
            output = self.output_layer(layer_input)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        
        return outputs, current_hidden


class WiredCfC(nn.Module):
    """
    带结构约束的 CfC (Wired CfC)
    
    通过神经架构搜索(NAS)找到的最优连接模式
    适合特定任务的结构化网络
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        wiring: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 如果没有提供 wiring，使用全连接
        if wiring is None:
            self.register_buffer(
                'wiring', 
                torch.ones(hidden_size, hidden_size)
            )
        else:
            self.register_buffer('wiring', wiring)
        
        # 带结构约束的权重
        self.recurrent_weight = nn.Parameter(
            torch.randn(hidden_size, hidden_size) * 0.1
        )
        
        self.cfc = CfC(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用 wiring 掩码的前向传播"""
        # 应用结构约束
        masked_weight = self.recurrent_weight * self.wiring
        
        # 使用 CfC 处理
        output, new_hidden = self.cfc(x, hidden)
        
        # 应用结构化的循环连接
        if hidden is not None:
            structured = torch.matmul(hidden, masked_weight.t())
            output = output + torch.tanh(structured)
        
        final_output = self.output_layer(output)
        
        return final_output, new_hidden
