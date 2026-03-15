"""
通用液态层接口

提供统一的 API 来使用 LTC 和 CfC
支持自动选择模式和混合架构
"""

import torch
import torch.nn as nn
from typing import Optional, Literal
from .ltc import LTC, LTCSequence
from .cfc import CfC, CfCSequence


class LiquidLayer(nn.Module):
    """
    通用液态神经网络层
    
    支持模式:
    - "ltc": Liquid Time-Constant
    - "cfc": Closed-form Continuous-time
    - "auto": 根据任务自动选择
    
    特点:
    - 统一的接口
    - 自动处理序列数据
    - 支持混合架构
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mode: Literal["ltc", "cfc", "auto"] = "auto",
        num_layers: int = 1,
        **kwargs
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.num_layers = num_layers
        
        # 自动选择模式
        if mode == "auto":
            # 对于长序列，CfC 更高效
            # 对于需要精细时间建模的，LTC 更灵活
            mode = "cfc"
        
        self.mode = mode
        
        # 创建对应的层
        if mode == "ltc":
            self.core = LTC(input_size, hidden_size, **kwargs)
        elif mode == "cfc":
            self.core = CfC(input_size, hidden_size, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def forward(self, x, hidden=None):
        """统一的前向接口"""
        return self.core(x, hidden)


class LiquidNetwork(nn.Module):
    """
    完整的液态神经网络
    
    用于时间序列预测、序列建模等任务
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        mode: Literal["ltc", "cfc", "auto"] = "cfc",
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 选择序列模型
        if mode == "ltc":
            self.rnn = LTCSequence(
                input_size, hidden_size, hidden_size, 
                num_layers, **kwargs
            )
        else:
            self.rnn = CfCSequence(
                input_size, hidden_size, hidden_size,
                num_layers, mode, **kwargs
            )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        direction_factor = 2 if bidirectional else 1
        self.output_layer = nn.Linear(
            hidden_size * direction_factor, 
            output_size
        )
    
    def forward(self, x, hidden=None):
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, input_size]
        Returns:
            output: [batch_size, seq_len, output_size]
        """
        # RNN 处理
        rnn_out, hidden = self.rnn(x, hidden)
        
        # Dropout
        rnn_out = self.dropout(rnn_out)
        
        # 输出层
        output = self.output_layer(rnn_out)
        
        return output, hidden
    
    def predict(self, x, steps: int = 1):
        """
        多步预测
        
        Args:
            x: 历史序列 [batch_size, seq_len, input_size]
            steps: 预测步数
        
        Returns:
            predictions: [batch_size, steps, output_size]
        """
        self.eval()
        with torch.no_grad():
            # 获取最后的状态
            _, hidden = self.forward(x)
            
            predictions = []
            last_input = x[:, -1:, :]  # [batch, 1, input]
            
            for _ in range(steps):
                # 预测下一步
                pred, hidden = self.forward(last_input, hidden)
                predictions.append(pred[:, -1, :])  # 取最后一个时间步
                
                # 使用预测作为下一步输入 (自回归)
                last_input = pred[:, -1:, :]
            
            predictions = torch.stack(predictions, dim=1)
            
        return predictions


class LiquidForecaster(nn.Module):
    """
    专门用于时间序列预测的液态网络
    
    特点:
    - 支持多步预测
    - 支持概率预测 (输出分布)
    - 内置归一化
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        forecast_horizon: int = 1,
        probabilistic: bool = False,
        mode: str = "cfc"
    ):
        super().__init__()
        
        self.input_size = input_size
        self.forecast_horizon = forecast_horizon
        self.probabilistic = probabilistic
        
        # 核心网络
        output_size = input_size * 2 if probabilistic else input_size
        
        self.network = LiquidNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            mode=mode,
            num_layers=num_layers
        )
        
        # 归一化参数 (运行时统计)
        self.register_buffer('mean', torch.zeros(input_size))
        self.register_buffer('std', torch.ones(input_size))
        self.register_buffer('initialized', torch.tensor(0))
    
    def normalize(self, x):
        """归一化输入"""
        if self.initialized:
            return (x - self.mean) / (self.std + 1e-8)
        return x
    
    def denormalize(self, x):
        """反归一化输出"""
        if self.initialized:
            return x * self.std + self.mean
        return x
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: [batch, seq_len, features]
        Returns:
            如果 probabilistic=False: [batch, seq_len, features]
            如果 probabilistic=True: (mean, std) 每个都是 [batch, seq_len, features]
        """
        x_norm = self.normalize(x)
        output, hidden = self.network(x_norm, hidden)
        
        if self.probabilistic:
            # 输出均值和标准差
            mean, log_std = torch.chunk(output, 2, dim=-1)
            std = torch.exp(log_std)
            return (mean, std), hidden
        else:
            return output, hidden
    
    def forecast(self, x, horizon: Optional[int] = None):
        """
        多步预测
        
        Args:
            x: 历史数据 [batch, seq_len, features]
            horizon: 预测步数 (默认使用 self.forecast_horizon)
        
        Returns:
            预测结果
        """
        if horizon is None:
            horizon = self.forecast_horizon
        
        self.eval()
        with torch.no_grad():
            predictions = self.network.predict(x, horizon)
            
            if self.probabilistic:
                mean, log_std = torch.chunk(predictions, 2, dim=-1)
                std = torch.exp(log_std)
                mean = self.denormalize(mean)
                std = std * self.std  # 调整标准差
                return mean, std
            else:
                return self.denormalize(predictions)
