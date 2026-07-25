import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LNGRU(nn.Module):
    """Layer Normalized GRU - 完全兼容nn.GRU的接口"""

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False):
        super(LNGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        if bidirectional:
            raise NotImplementedError("Bidirectional LNGRU not implemented yet")

        # 为每一层创建参数
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.layers.append(LNGRUCell(layer_input_size, hidden_size, bias))

        # Dropout层
        if dropout > 0 and num_layers > 1:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None

    def forward(self, input, hx=None):
        """
        前向传播 - 完全兼容nn.GRU接口

        Args:
            input: (seq_len, batch, input_size) 或 (batch, seq_len, input_size)
            hx: (num_layers, batch, hidden_size) 或 None

        Returns:
            output: (seq_len, batch, hidden_size) 或 (batch, seq_len, hidden_size)
            hn: (num_layers, batch, hidden_size)
        """
        if self.batch_first:
            input = input.transpose(0, 1)  # (batch, seq_len, input_size) -> (seq_len, batch, input_size)

        seq_len, batch_size = input.size(0), input.size(1)

        # 初始化隐藏状态
        if hx is None:
            hx = input.new_zeros(self.num_layers, batch_size, self.hidden_size)

        # 存储每层的隐藏状态
        h_n = []
        layer_output = input

        # 逐层处理
        for layer_idx in range(self.num_layers):
            layer_h = hx[layer_idx]  # (batch, hidden_size)
            layer_outputs = []

            # 逐时间步处理
            for t in range(seq_len):
                layer_h = self.layers[layer_idx](layer_output[t], layer_h)
                layer_outputs.append(layer_h)

            # 组合该层的所有时间步输出
            layer_output = torch.stack(layer_outputs, dim=0)  # (seq_len, batch, hidden_size)
            h_n.append(layer_h)

            # 在层之间应用dropout（除了最后一层）
            if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)

        # 最终输出
        output = layer_output
        hn = torch.stack(h_n, dim=0)  # (num_layers, batch, hidden_size)

        if self.batch_first:
            output = output.transpose(0, 1)  # (seq_len, batch, hidden_size) -> (batch, seq_len, hidden_size)

        return output, hn


class LNGRUCell(nn.Module):
    """单个LNGRU Cell"""

    def __init__(self, input_size, hidden_size, bias=True):
        super(LNGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # GRU的线性变换参数
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))

        if bias:
            self.bias_ih = nn.Parameter(torch.randn(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.randn(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        # 层归一化 - 关键改进
        self.ln_reset = nn.LayerNorm(hidden_size)
        self.ln_update = nn.LayerNorm(hidden_size)
        self.ln_new = nn.LayerNorm(hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.uniform_(weight, -std, std)
            else:
                nn.init.zeros_(weight)

    def forward(self, input, hx):
        """
        LNGRU Cell前向传播

        Args:
            input: (batch, input_size)
            hx: (batch, hidden_size)

        Returns:
            new_h: (batch, hidden_size)
        """
        # 线性变换
        gi = F.linear(input, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)

        # 分割为reset, update, new gates
        i_r, i_u, i_n = gi.chunk(3, 1)
        h_r, h_u, h_n = gh.chunk(3, 1)

        # Reset gate - 添加层归一化
        r = torch.sigmoid(self.ln_reset(i_r + h_r))

        # Update gate - 添加层归一化
        u = torch.sigmoid(self.ln_update(i_u + h_u))

        # New gate - 添加层归一化
        n = torch.tanh(self.ln_new(i_n + r * h_n))

        # 计算新的隐藏状态
        new_h = (1 - u) * n + u * hx

        return new_h


