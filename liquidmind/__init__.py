"""
LiquidMind - 液态神经网络实现
动态适应的智能系统

基于 MIT 的 Liquid Time-Constant (LTC) 和 Closed-form Continuous-time (CfC) 网络
"""

__version__ = "0.1.0"
__author__ = "EC"

from .ltc import LTC, LTCSequence
from .cfc import CfC, CfCSequence
from .liquid_layer import LiquidLayer, LiquidNetwork, LiquidForecaster

__all__ = ['LTC', 'LTCSequence', 'CfC', 'CfCSequence', 'LiquidLayer', 'LiquidNetwork', 'LiquidForecaster']
