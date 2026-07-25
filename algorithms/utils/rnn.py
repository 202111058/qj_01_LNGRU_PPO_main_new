import torch
import torch.nn as nn


# 版本一：基于 torch.nn.GRU 的高效 LNGRU 实现
class EfficientLNGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False, use_orthogonal=False):
        super(EfficientLNGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size * self.num_directions
            self.gru_layers.append(
                nn.GRU(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    bias=bias,
                    batch_first=batch_first,
                    dropout=0,
                    bidirectional=bidirectional
                )
            )

        self.ln_input = nn.ModuleList([
            nn.LayerNorm(input_size if i == 0 else hidden_size * self.num_directions)
            for i in range(num_layers)
        ])

        if dropout > 0 and num_layers > 1:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.ln_output = nn.LayerNorm(hidden_size * self.num_directions)
        self._init_weights(use_orthogonal)

    def _init_weights(self, use_orthogonal):
        if not use_orthogonal:
            return
        for gru in self.gru_layers:
            for name, param in gru.named_parameters():
                if 'weight_ih' in name:
                    nn.init.orthogonal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

    def forward(self, x, hx=None):
        if self.batch_first and x.dim() == 3:
            # 输入已经是 (N, T, D), GRU 会正确处理
            pass
        elif not self.batch_first and x.dim() == 3:
            x = x.transpose(0, 1)

        if hx is None:
            batch_size = x.size(0) if self.batch_first else x.size(1)
            hx = torch.zeros(self.num_layers * self.num_directions,
                             batch_size, self.hidden_size,
                             device=x.device, dtype=x.dtype)

        output = x
        final_hidden_states = []

        for i, gru_layer in enumerate(self.gru_layers):
            normalized_input = self.ln_input[i](output)
            layer_hx = hx[i * self.num_directions:(i + 1) * self.num_directions]
            output, layer_h_n = gru_layer(normalized_input, layer_hx)

            if self.dropout is not None and i < self.num_layers - 1:
                output = self.dropout(output)

            final_hidden_states.append(layer_h_n)

        output = self.ln_output(output)
        h_n = torch.cat(final_hidden_states, dim=0)

        return output, h_n


class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()
        self.rnn = EfficientLNGRU(
            input_size=inputs_dim,
            hidden_size=outputs_dim,
            num_layers=recurrent_N,
            use_orthogonal=use_orthogonal,
            batch_first=True
        )

    def forward(self, x, hxs, masks):
        # [关键修复] 转换 hxs 形状以匹配 PyTorch GRU 的要求，并确保内存连续
        # hxs 输入形状是 (N, L, H)，N=批次, L=层数
        # PyTorch GRU 要求 (L, N, H)，且内存连续
        hxs = hxs.transpose(0, 1).contiguous()

        # 现在 hxs 是 (L, N, H)，我们可以安全地获取批次大小 N
        N = hxs.size(1)

        if x.size(0) == N:
            # 单个时间步 (N, features) -> (N, 1, features)
            x = x.unsqueeze(1)
        else:
            # 多个时间步 (T*N, features) -> (N, T, features)
            T = x.size(0) // N
            x = x.view(N, T, -1)

        # self.rnn 接收的 hxs 现在是正确的 (L, N, H) 形状且内存连续
        x, hxs = self.rnn(x, hxs)

        # [关键修复] 将 hxs 转换回项目中约定的 (N, L, H) 格式，并确保内存连续
        hxs = hxs.transpose(0, 1).contiguous()

        x = x.reshape(-1, x.size(-1))
        return x, hxs