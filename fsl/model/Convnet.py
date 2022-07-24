# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class Convnet(nn.Module):
    def __init__(self, config):
        super(Convnet, self).__init__()
        self.config = config

        self.input_size = [1, 28, 28]

        dim_cnn_out = config.dim_cnn_out
        filter_num = config.num_filter
        filter_sizes = [int(fsz) for fsz in config.filter_sizes.split(',')]

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, fsz)) for fsz in filter_sizes])

        self.dropout = nn.Dropout(config.dropout)

        self.linear = nn.Linear(len(filter_sizes) * filter_num, dim_cnn_out)
        if self.config.mode == 'train-test' or self.config.mode == 'cross validation':
            label_num = config.num_class
            if self.config.output_extend == 'pretrain':
                self.classifier_pretrain = nn.Linear(dim_cnn_out,
                                                     label_num)  # label_num: 24 + 1 (number of meta-train classes + 1 random sequence class)
            elif self.config.output_extend == 'finetune':
                self.classifier_finetune = nn.Linear(dim_cnn_out,
                                                     label_num)  # label_num: 2 for binary peptide classification
            else:
                print('Error, No Such Output Extend')

    def forward(self, x):
        # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大=长度
        # print('raw x', x.size())
        x = x.float()
        # 经过view函数x的维度变为(batch_size, input_chanel=1, w=max_len, h=embedding_dim)
        x = x.view(x.size(0), 1, self.input_size[1], self.input_size[2])
        # print('view x', x.size())

        # 经过卷积运算,x中每个运算结果维度为(batch_size, out_chanel, w, h=1)
        x = [F.relu(conv(x)) for conv in self.convs]
        # print('conv x', len(x), [x_item.size() for x_item in x])

        # 经过最大池化层,维度变为(batch_size, out_chanel, w=1, h=1)
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        # print('max_pool2d x', len(x), [x_item.size() for x_item in x])

        # 将不同卷积核运算结果维度（batch，out_chanel,w,h=1）展平为（batch, outchanel*w*h）
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        # print('flatten x', len(x), [x_item.size() for x_item in x])

        # 将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
        x = torch.cat(x, 1)
        # print('concat x', x.size())

        # dropout层
        x = self.dropout(x)
        # print('dropout x', x.size())

        # 全连接层
        output = self.linear(x)

        # print('fully connection', output.size())

        if self.config.mode == 'train-test' or self.config.mode == 'cross validation':
            if self.config.output_extend == 'pretrain':
                output = self.classifier_pretrain(output)
            elif self.config.output_extend == 'finetune':
                output = self.classifier_finetune(output)
        return output
