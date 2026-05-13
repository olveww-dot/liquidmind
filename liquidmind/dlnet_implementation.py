"""
DLNet: Dual-Stage Distillation of Liquid Neural Networks
=========================================================
参考: arXiv:2601.06227 - "When Smaller Wins"

核心创新:
1. Euler离散化 → 嵌入式兼容
2. 双阶段知识蒸馏 → 时序行为迁移
3. Pareto引导压缩 → 精度/效率平衡

目标: 将大模型压缩到边缘设备 (Arduino 94KB, 21ms)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import math

# ============================================================
# 第一阶段: Euler离散化的LTC
# ============================================================

class EulerDiscreteLTC(nn.Module):
    """
    使用Euler前向离散化的LTC
    
    原始ODE: dh/dt = -h/τ + W·σ(x) + b
    Euler离散: h[t+1] = h[t] + dt * (-h[t]/τ + W·σ(x))
    
    优势: 
    - 无需ODE求解器
    - 固定时间步长，易于硬件实现
    - 内存效率高
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int,
        dt: float = 0.1,      # Euler步长
        tau_min: float = 1.0, # τ下限
        tau_max: float = 10.0, # τ上限
        bias: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        
        # 输入到隐藏层
        self.W_in = nn.Linear(input_size, hidden_size, bias=bias)
        
        # 循环连接
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 时间常数网络 (输入依赖的τ)
        self.tau_net = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # 初始化
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_rec.weight)
    
    def get_tau(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """计算输入依赖的时间常数 τ(x, h)"""
        concat = torch.cat([x, h], dim=-1)
        tau_hidden = self.tau_net(concat)
        # 映射到 [tau_min, tau_max]
        return self.tau_min + (self.tau_max - self.tau_min) * tau_hidden
    
    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)
            h0: (batch, hidden_size), 初始隐藏状态
        
        Returns:
            outputs: (batch, seq_len, hidden_size)
            final_h: (batch, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h0
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_size)
            
            # 计算 τ(x, h)
            tau = self.get_tau(x_t, h)  # (batch, hidden_size)
            
            # 预激活值
            pre_act = self.W_in(x_t) + self.W_rec(h)
            
            # Euler更新
            dh = (-h / tau + torch.tanh(pre_act)) * self.dt
            h = h + dh
            
            outputs.append(h)
        
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_size)
        return outputs, h


class ProjectionLayer(nn.Module):
    """
    投影层: 将不同维度的隐状态映射到同一空间
    
    用于知识蒸馏时，学生和老师维度不同时
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ============================================================
# 第二阶段: 知识蒸馏
# ============================================================

class TemporalDistillationLoss(nn.Module):
    """
    时序知识蒸馏损失
    
    论文核心: 双阶段蒸馏
    Stage 1: 让学生学习老师的"软标签"时序输出 (通过投影层对齐)
    Stage 2: 匹配最终状态
    
    修复: 使用投影层处理不同维度
    """
    
    def __init__(
        self, 
        teacher_hidden: int,
        student_hidden: int,
        temperature: float = 2.0
    ):
        super().__init__()
        self.temperature = temperature
        
        # 投影层: 将学生/老师的隐藏状态映射到统一维度
        self.proj_student = nn.Linear(student_hidden, min(student_hidden, teacher_hidden))
        self.proj_teacher = nn.Linear(teacher_hidden, min(student_hidden, teacher_hidden))
    
    def forward(
        self, 
        student_h: torch.Tensor,   # (batch, seq_len, student_hidden)
        teacher_h: torch.Tensor,   # (batch, seq_len, teacher_hidden)
        student_final: torch.Tensor, # (batch, student_hidden)
        teacher_final: torch.Tensor, # (batch, teacher_hidden)
    ) -> dict:
        """
        计算蒸馏损失
        """
        # 投影到统一维度
        student_proj = self.proj_student(student_h)
        teacher_proj = self.proj_teacher(teacher_h)
        
        # 1. 时序轨迹匹配
        soft_loss = F.mse_loss(student_proj, teacher_proj)
        
        # 2. 最终状态匹配
        student_final_proj = self.proj_student(student_final)
        teacher_final_proj = self.proj_teacher(teacher_final)
        final_loss = F.mse_loss(student_final_proj, teacher_final_proj)
        
        # 3. 状态差异的统计量 (均值、标准差)
        student_mean = student_h.mean(dim=[1, 2])
        teacher_mean = teacher_h.mean(dim=[1, 2])
        mean_loss = F.mse_loss(student_mean, teacher_mean)
        
        return {
            'soft_loss': soft_loss.item(),
            'final_loss': final_loss.item(),
            'mean_loss': mean_loss.item(),
            'total': soft_loss + final_loss + 0.1 * mean_loss
        }


# ============================================================
# 第三阶段: Pareto最优压缩
# ============================================================

class ParetoCompressor:
    """
    Pareto引导的模型压缩
    
    目标: 在 预测误差 和 模型大小 两个目标之间找到最优权衡
    方法: 训练多个不同压缩率的模型，选择Pareto前沿上的模型
    """
    
    def __init__(
        self,
        compression_ratios: List[float] = [0.25, 0.5, 0.75],
    ):
        self.compression_ratios = compression_ratios
    
    def create_student(
        self, 
        teacher: EulerDiscreteLTC,
        compression_ratio: float
    ) -> EulerDiscreteLTC:
        """
        创建压缩后的学生模型
        """
        new_hidden = max(4, int(teacher.hidden_size * compression_ratio))
        
        student = EulerDiscreteLTC(
            input_size=teacher.input_size,
            hidden_size=new_hidden,
            dt=teacher.dt,
            tau_min=teacher.tau_min,
            tau_max=teacher.tau_max
        )
        
        return student
    
    @staticmethod
    def is_pareto_efficient(costs: np.ndarray) -> List[bool]:
        """
        找出Pareto前沿上的点
        
        Args:
            costs: (n_points, 2) - 越小越好的二维成本
        
        Returns:
            is_efficient: 每个点是否Pareto最优
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # 移除被当前点支配的点
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
                is_efficient[i] = True
        return is_efficient.tolist()


# ============================================================
# 完整蒸馏流程
# ============================================================

class DLNetDistiller:
    """
    DLNet蒸馏器
    
    完整流程:
    1. 训练大模型 (Teacher)
    2. 训练小模型 (Student) 蒸馏
    3. Pareto选择
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        teacher_hidden: int = 64,
        student_hidden: int = 16,
        dt: float = 0.1
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.dt = dt
        
        # 教师模型 (大)
        self.teacher_ltc = EulerDiscreteLTC(input_size, teacher_hidden, dt=dt)
        self.teacher_head = nn.Linear(teacher_hidden, output_size)
        
        # 学生模型 (小)
        self.student_ltc = EulerDiscreteLTC(input_size, student_hidden, dt=dt)
        self.student_head = nn.Linear(student_hidden, output_size)
        
        # 蒸馏损失 (带投影层)
        self.distill_loss = TemporalDistillationLoss(
            teacher_hidden=teacher_hidden,
            student_hidden=student_hidden
        )
        
        # 损失函数
        self.output_loss = nn.MSELoss()
        
        # 优化器
        self.teacher_optimizer = None
        self.student_optimizer = None
        
        self._setup_optimizers()
    
    def _setup_optimizers(self):
        self.teacher_optimizer = torch.optim.Adam(
            list(self.teacher_ltc.parameters()) + list(self.teacher_head.parameters()),
            lr=0.001
        )
        self.student_optimizer = torch.optim.Adam(
            list(self.student_ltc.parameters()) + list(self.student_head.parameters()),
            lr=0.001
        )
    
    def train_teacher(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor,
        epochs: int = 100
    ) -> List[float]:
        """阶段1: 训练教师模型"""
        losses = []
        
        for epoch in range(epochs):
            self.teacher_optimizer.zero_grad()
            
            # 前向
            h, _ = self.teacher_ltc(X)  # LTC层
            pred = self.teacher_head(h[:, -1, :])  # 预测
            
            loss = self.output_loss(pred, y)
            loss.backward()
            self.teacher_optimizer.step()
            
            losses.append(loss.item())
        
        return losses
    
    def distill(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 100
    ) -> dict:
        """阶段2: 蒸馏训练"""
        losses = {'total': [], 'output': [], 'distill': []}
        
        for epoch in range(epochs):
            self.student_optimizer.zero_grad()
            
            # 教师前向 (不更新)
            with torch.no_grad():
                teacher_h, teacher_final = self.teacher_ltc(X)
                teacher_pred = self.teacher_head(teacher_h[:, -1, :])
            
            # 学生前向
            student_h, student_final = self.student_ltc(X)
            student_pred = self.student_head(student_h[:, -1, :])
            
            # 蒸馏损失 (隐状态对齐)
            distill_dict = self.distill_loss(
                student_h, teacher_h,
                student_final, teacher_final
            )
            
            # 输出损失
            output_loss = self.output_loss(student_pred, y)
            
            # 总损失 = 输出损失 + 蒸馏损失
            total_loss = output_loss + 0.5 * distill_dict['total']
            
            total_loss.backward()
            self.student_optimizer.step()
            
            losses['total'].append(total_loss.item())
            losses['output'].append(output_loss.item())
            losses['distill'].append(distill_dict['total'])
        
        return losses
    
    def get_model_size(self, params: List[nn.Parameter]) -> int:
        """获取参数量"""
        return sum(p.numel() for p in params if p.requires_grad)
    
    def summary(self):
        """打印模型摘要"""
        teacher_params = list(self.teacher_ltc.parameters()) + list(self.teacher_head.parameters())
        student_params = list(self.student_ltc.parameters()) + list(self.student_head.parameters())
        
        print("=" * 50)
        print("DLNet 模型摘要")
        print("=" * 50)
        print(f"教师参数: {self.get_model_size(teacher_params):,}")
        print(f"学生参数: {self.get_model_size(student_params):,}")
        print(f"压缩比: {self.get_model_size(teacher_params)/self.get_model_size(student_params):.1f}x")


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DLNet: 双阶段蒸馏 LNN 实现测试")
    print("=" * 60)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 生成测试数据 (电池循环数据模拟)
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 500
    seq_len = 50
    input_size = 5  # 电压, 电流, 温度, 阻抗, 循环数
    output_size = 1  # SOH (State of Health)
    
    # 生成更真实的数据: SOH随循环衰减
    X = []
    y = []
    for i in range(n_samples):
        # 模拟电池衰减轨迹
        cycle = i / n_samples  # 0 到 1
        soh = 1.0 - 0.3 * cycle + 0.1 * np.random.randn()  # SOH 从1.0衰减到0.7
        
        # 生成序列特征
        seq = np.random.randn(seq_len, input_size) * 0.1
        seq[:, 0] = 4.2 - 0.2 * cycle + 0.01 * np.random.randn()  # 电压衰减
        seq[:, 1] = 1.0 + 0.1 * np.random.randn()  # 电流
        seq[:, 2] = 25 + 5 * cycle + 1 * np.random.randn()  # 温度上升
        seq[:, 3] = 0.1 + 0.05 * cycle + 0.01 * np.random.randn()  # 阻抗增加
        seq[:, 4] = np.linspace(0, cycle, seq_len)  # 循环进度
        
        X.append(seq)
        y.append(soh)
    
    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y)).reshape(-1, 1)
    
    print(f"\n数据: X={X.shape}, y={y.shape}")
    print(f"SOH范围: [{y.min().item():.3f}, {y.max().item():.3f}]")
    
    # 创建蒸馏器
    distiller = DLNetDistiller(
        input_size=input_size,
        output_size=output_size,
        teacher_hidden=32,
        student_hidden=8,
        dt=0.1
    )
    distiller.summary()
    
    # 阶段1: 训练教师
    print("\n" + "=" * 50)
    print("阶段1: 训练教师模型")
    print("=" * 50)
    teacher_losses = distiller.train_teacher(X, y, epochs=100)
    print(f"教师最终损失: {teacher_losses[-1]:.6f}")
    
    # 阶段2: 蒸馏
    print("\n" + "=" * 50)
    print("阶段2: 学生模型蒸馏")
    print("=" * 50)
    distill_losses = distiller.distill(X, y, epochs=100)
    print(f"学生最终损失: {distill_losses['total'][-1]:.6f}")
    
    # 最终评估
    print("\n" + "=" * 50)
    print("最终结果")
    print("=" * 50)
    
    with torch.no_grad():
        # 教师预测
        t_h, _ = distiller.teacher_ltc(X[:20])
        t_pred = distiller.teacher_head(t_h[:, -1, :])
        
        # 学生预测
        s_h, _ = distiller.student_ltc(X[:20])
        s_pred = distiller.student_head(s_h[:, -1, :])
        
        teacher_mse = F.mse_loss(t_pred, y[:20]).item()
        student_mse = F.mse_loss(s_pred, y[:20]).item()
        
        print(f"\n教师 MSE: {teacher_mse:.6f}")
        print(f"学生 MSE: {student_mse:.6f}")
        print(f"\n学生/教师 比率: {student_mse/max(teacher_mse, 1e-10):.2f}x")
        
        # 计算参数效率
        teacher_params = sum(p.numel() for p in distiller.teacher_ltc.parameters()) + sum(p.numel() for p in distiller.teacher_head.parameters())
        student_params = sum(p.numel() for p in distiller.student_ltc.parameters()) + sum(p.numel() for p in distiller.student_head.parameters())
        
        print(f"\n参数量:")
        print(f"  教师: {teacher_params:,}")
        print(f"  学生: {student_params:,}")
        print(f"  压缩比: {teacher_params/student_params:.1f}x")
        
        # 估算Arduino部署大小
        # 假设 float32 -> int8 量化 (4x压缩)
        estimated_kb = student_params * 4 / 1024
        print(f"\nArduino部署估计: ~{estimated_kb:.1f} KB (int8量化后)")
    
    print("\n" + "=" * 50)
    print("✅ DLNet 实现测试完成!")
    print("=" * 50)
