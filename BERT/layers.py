# coding=utf-8
import torch.nn.functional as F
import torch.nn as nn
import torch


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes):
        super(Conv1d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.init_params()

    def init_params(self):
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):
        return [F.relu(conv(x)) for conv in self.convs]


class Embedding(nn.Module):
    """
    word and char embedding

    Input shape: word_emb=(batch_size,sentence_length,emb_size) char_emb=(batch_size,sentence_length,word_length,emb_size)
    Output shape: y= (batch_size,sentence_length,word_emb_size+char_emb_size)
    """

    def __init__(self, highway_layers, word_dim, char_dim):
        super(Embedding, self).__init__()
        self.highway = Highway(highway_layers, word_dim + char_dim)

    def forward(self, word_emb, char_emb):
        char_emb, _ = torch.max(char_emb, 2)

        emb = torch.cat([word_emb, char_emb], dim=2)
        emb = self.highway(emb)

        return emb


class Highway(nn.Module):
    """
    Input shape=(batch_size,dim,dim)
    Output shape=(batch_size,dim,dim)
    """

    def __init__(self, layer_num, dim=600):
        super(Highway, self).__init__()
        self.layer_num = layer_num
        self.linear = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(self.layer_num)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim)
                                   for _ in range(self.layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return x


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dropout):
        """
        Args:
            input_size: x 的特征维度
            hidden_size: 隐层的特征维度
            num_layers: LSTM 层数
        """
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional,
            dropout=dropout
        )

        self.init_params()

    def init_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}'))
            nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}'))
            nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}'), val=0)
            nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}'), val=0)
            getattr(self.rnn, f'bias_hh_l{i}').chunk(4)[1].fill_(1)

            if self.rnn.bidirectional:
                nn.init.orthogonal_(
                    getattr(self.rnn, f'weight_hh_l{i}_reverse'))
                nn.init.kaiming_normal_(
                    getattr(self.rnn, f'weight_ih_l{i}_reverse'))
                nn.init.constant_(
                    getattr(self.rnn, f'bias_hh_l{i}_reverse'), val=0)
                nn.init.constant_(
                    getattr(self.rnn, f'bias_ih_l{i}_reverse'), val=0)
                getattr(self.rnn, f'bias_hh_l{i}_reverse').chunk(4)[1].fill_(1)

    def forward(self, x, lengths):
        # x: [seq_len, batch_size, input_size]
        # lengths: [batch_size]
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # packed_x， packed_output: PackedSequence 对象
        # hidden: [num_layers * bidirectional, batch_size, hidden_size]
        # cell: [num_layers * bidirectional, batch_size, hidden_size]
        packed_output, (hidden, cell) = self.rnn(packed_x)

        # output: [real_seq_len, batch_size, hidden_size * 2]
        # output_lengths: [batch_size]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        return hidden, output
