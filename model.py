import torch
import torch.nn as nn
import numpy as np
import math


def linear_weights_init(m):
    if isinstance(m, nn.Linear):
        stdv = 1. / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        return self.pe[:seq_len, :]


class STTransformer(nn.Module):
    ''' Spatial Temporal Transformer
        local_attention: spatial encoder
        global_attention: temporal decoder
        position_embedding: frame encoding (window_size*dim)    
    '''

    def __init__(self, node_dim, feature_dim, time_window, predicted_len, spatial_hidden_dim=128, temporal_hidden_dim=128, spatial_enc_layer_num=1, temporal_enc_layer_num=1, nhead=2, dim_feedforward=2048,
                 dropout=0.1):
        super(STTransformer, self).__init__()

        spatial_encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.spatial_attention = nn.TransformerEncoder(
            spatial_encoder_layer, spatial_enc_layer_num)
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim*2, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.temporal_attention = nn.TransformerEncoder(
            temporal_encoder_layer, temporal_enc_layer_num)
        self.position_encoder = PositionalEncoding(
            2*feature_dim)  # timestamp encoding
        self.global_info = nn.Sequential(
            nn.Linear(node_dim*feature_dim, spatial_hidden_dim),
            nn.ReLU(),
            nn.Linear(spatial_hidden_dim, feature_dim),
        )
        self.global_info.apply(linear_weights_init)

        self.predictor = nn.Sequential(
            nn.Linear(time_window*2*feature_dim, temporal_hidden_dim),
            nn.ReLU(),
            nn.Linear(temporal_hidden_dim, predicted_len),
        )
        self.predictor.apply(linear_weights_init)

    def forward(self, node_t, nodes_st):
        # pytorch 1.1.8 does not support batch_first in Transformer, so we
        # need to permute to have batch first
        # input node_t: B*T*F, nodes_st: B*N*T*F
        nodes_st = nodes_st.permute(0, 2, 1, 3)
        # node_t: B*T*F (batch*time_window*feature_dim)
        # nodes_st: B*T*N*F (batch*time_window*node_dim*feature_dim)
        batch_size = node_t.shape[0]
        time_window = node_t.shape[1]
        node_t_global_list = []
        for t in range(time_window):
            # spatial attention
            # nodes_s: B*N*F (batch*node_dim*feature_dim)
            nodes_s = torch.squeeze(nodes_st[:, t, :, :], dim=1)
            nodes_s = nodes_s.permute(1, 0, 2)
            spatial_output = self.spatial_attention(nodes_s)
            spatial_output = spatial_output.permute(1, 0, 2)
            spatial_output = spatial_output.contiguous().view(batch_size, -1)
            global_s_info = self.global_info(spatial_output)  # B*F
            local_s_info = torch.stack(
                (node_t[:, t, :], global_s_info)).view(batch_size, -1)  # B*2F
            node_t_global_list.append(local_s_info)
        node_t_global = torch.stack(
            node_t_global_list).permute(1, 0, 2)  # B*T*2F

        # temporal attention
        temporal_encoding = self.position_encoder(
            time_window).unsqueeze(0).tile(batch_size, 1, 1)  # 1*T*2F
        temporal_input = node_t_global+temporal_encoding
        # current transformer does not support batch first
        temporal_input = temporal_input.permute(1, 0, 2)
        temporal_output = self.temporal_attention(temporal_input)
        temporal_output = temporal_output.permute(
            1, 0, 2).contiguous().view(batch_size, -1)
        output = self.predictor(temporal_output)
        return output


class LSTM_M(nn.Module):
    def __init__(self,  feature_dim, predicted_len, temporal_hidden_dim=128):
        super(LSTM_M, self).__init__()
        self.predictor = nn.LSTM(
            feature_dim, temporal_hidden_dim, 1, batch_first=True)
        self.predictor.apply(linear_weights_init)
        self.output = nn.Sequential(
            nn.Linear(temporal_hidden_dim, predicted_len)
        )
        self.output.apply(linear_weights_init)
        self.temporal_hidden_dim = temporal_hidden_dim

    def forward(self, node_t):
        batch_size = node_t.shape[0]
        h0 = torch.randn(1, batch_size, self.temporal_hidden_dim).cuda()
        c0 = torch.randn(1, batch_size, self.temporal_hidden_dim).cuda()
        hidden_t = (h0, c0)
        output, hidden_t = self.predictor(node_t, hidden_t)
        output = output[:, -1, :]
        output = self.output(output)
        return output


class LR(nn.Module):
    def __init__(self, feature_dim, time_window, predicted_len):
        super(LR, self).__init__()
        self.f1 = torch.nn.Sequential(
            torch.nn.Linear(time_window*feature_dim, predicted_len),
            nn.Sigmoid(),
        )

    def forward(self, node_t):
        output = self.f1(node_t)
        return output
