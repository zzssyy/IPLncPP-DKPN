import torch.nn as nn
import torch
import math
#
# class LSTM(nn.Module):
#     def __init__(self, config):
#         super(LSTM, self).__init__()
#         device = torch.device('cuda:0')
#         self.config = config
#         self.input_size = [1, 28, 28]
#         hidden_layer_size = config.hidden_layer_size
#
#         # 创建LSTM层和linear层，LSTM层提取特征，linear层用作最后的预测
#         # LSTM算法接受三个输入：先前的隐藏状态，先前的单元状态和当前输入。
#         self.lstm = nn.LSTM(self.input_size[1]*self.input_size[2], hidden_layer_size)
#
#         #初始化隐含状态及细胞状态C，hidden_cell变量包含先前的隐藏状态和单元状态
#         self.hidden_cell = (torch.zeros(1, 1, hidden_layer_size).to(device),
#                             torch.zeros(1, 1, hidden_layer_size).to(device))
#
#         if self.config.mode == 'train-test' or self.config.mode == 'cross validation':
#             output_size = config.num_class
#             if self.config.output_extend == 'pretrain':
#                 self.classifier_pretrain = nn.Linear(hidden_layer_size,
#                                                      output_size)  # label_num: 24 + 1 (number of meta-train classes + 1 random sequence class)
#             elif self.config.output_extend == 'finetune':
#                 self.classifier_finetune = nn.Linear(hidden_layer_size,
#                                                      output_size)  # label_num: 2 for binary peptide classification
#             else:
#                 print('Error, No Such Output Extend')
#
#     def forward(self, input_seq):
#         input_seq = input_seq.float()
#         print('view x', input_seq.size())
#         input_seq = input_seq.view(len(input_seq), 1, -1)
#         print('view x', input_seq.size())
#         #lstm的输出是当前时间步的隐藏状态ht和单元状态ct以及输出lstm_out
#         lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
#         #按照lstm的格式修改input_seq的形状，作为linear层的输入
#         output = lstm_out[:, -1, :]
#         print("output", output.size())
#
#         if self.config.mode == 'train-test' or self.config.mode == 'cross validation':
#             output = self.linear(lstm_out.view(len(input_seq), -1))
#             # 返回predictions的最后一个元素
#             if self.config.output_extend == 'pretrain':
#                 output = output[-1]
#             elif self.config.output_extend == 'finetune':
#                 output = output[-1]
#         return output


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        # device = torch.device('cuda:0')
        self.config = config
        self.input_size = [1, 28, 28]
        self.hidden_layer_size = config.hidden_layer_size
        # self.W = nn.Parameter(torch.Tensor(self.input_size[1]*self.input_size[2], self.hidden_layer_size * 4))
        self.W = nn.Parameter(torch.Tensor(self.input_size[1], self.hidden_layer_size * 4))
        self.U = nn.Parameter(torch.Tensor(self.hidden_layer_size, self.hidden_layer_size * 4))
        self.bias = nn.Parameter(torch.Tensor(self.hidden_layer_size * 4))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_layer_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

        if self.config.mode == 'train-test' or self.config.mode == 'cross validation':
            output_size = self.config.num_class
            if self.config.output_extend == 'pretrain':
                self.classifier_pretrain = nn.Linear(self.hidden_layer_size,
                                                         output_size)  # label_num: 24 + 1 (number of meta-train classes + 1 random sequence class)
            elif self.config.output_extend == 'finetune':
                self.classifier_finetune = nn.Linear(self.hidden_layer_size,
                                                         output_size)  # label_num: 2 for binary peptide classification
            else:
                print('Error, No Such Output Extend')


    def forward(self, x,
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        x = x.float()
        print(x.size())
        # x = x.view(x.size(0), 1, -1)
        x = x.view(x.size(0), 28, 28)
        # print(x.size())
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_layer_size).to(x.device),
                        torch.zeros(bs, self.hidden_layer_size).to(x.device))
        else:
            h_t, c_t = init_states

        HS = self.hidden_layer_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # print(x_t.size())
            # print(h_t.size())
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        # return hidden_seq, (h_t, c_t)
        output = hidden_seq[:, -1, :]
        print("output", output.size())

        if self.config.mode == 'train-test' or self.config.mode == 'cross validation':
            output = self.linear(hidden_seq.view(len(x), -1))
            # 返回predictions的最后一个元素
            if self.config.output_extend == 'pretrain':
                    output = output[-1]
            elif self.config.output_extend == 'finetune':
                    output = output[-1]
        return output