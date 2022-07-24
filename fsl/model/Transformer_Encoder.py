import torch
import torch.nn as nn
import numpy as np
import pickle
from fsl.config import config_meta
from fsl.util import util_model
import copy
import math
#
#
# # 目的是构造出一个注意力判断矩阵，一个[batch_size, seq_len, seq_len]的张量
# # 其中参与注意力计算的位置被标记为FALSE，将token为[PAD]的位置掩模标记为TRUE
# def get_attn_pad_mask(seq):
#     # 在BERT中由于是self_attention，seq_q和seq_k内容相同
#     # print('-' * 50, '掩模', '-' * 50)
#
#     batch_size, seq_len = seq.size()
#     # print('batch_size', batch_size)
#     # print('seq_len', seq_len)
#
#     # print('-' * 10, 'test', '-' * 10)
#     # print(seq_q.data.shape)
#     # print(seq_q.data.eq(0).shape)
#     # print(seq_q.data.eq(0).unsqueeze(1).shape)
#
#     # seq_q.data取出张量seq_q的数据
#     # seq_q.data.eq(0)是一个和seq_q.data相同shape的张量，seq_q.data对应位置为0时，结果的对应位置为TRUE，否则为FALSE
#     # eq(zero) is PAD token 如果等于0，证明那个位置是[PAD]，因此要掩模，计算自注意力时不需要考虑该位置
#     # unsqueeze(1)是在维度1处插入一个维度，维度1及其之后的维度往后移，从原来的[batch_size, seq_len]变成[batch_size, 1, seq_len]
#     pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
#     print("pad_attn_mask", pad_attn_mask.size())
#
#     # expand是将某一个shape为1的维度复制为自己想要的数量，这里是从[batch_size, 1, seq_len]将1维度复制seq_len份
#     # 结果是变成[batch_size, seq_len, seq_len]
#     pad_attn_mask_expand = pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
#
#     return pad_attn_mask_expand
#
#
# # 嵌入层
# class Embedding(nn.Module):
#     def __init__(self):
#         super(Embedding, self).__init__()
#         self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding (look-up table)
#         self.pos_embed = nn.Embedding(max_len, d_model)  # position embedding
#         # self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
#         self.norm = nn.LayerNorm(d_model)
#
#         # 改变初始化权重测试
#         # look_up_table = torch.rand([vocab_size, d_model], dtype=torch.float)
#         # look_up_table = torch.zeros([vocab_size, d_model], dtype=torch.float)
#         # print('old_look_up_table', look_up_table)
#         # torch.nn.init.uniform_(look_up_table, a=0, b=1)
#         # torch.nn.init.normal_(look_up_table)
#         # print('new_look_up_table', look_up_table)
#         # self.tok_embed = self.tok_embed.from_pretrained(look_up_table)
#         # print('self.tok_embed.weight', self.tok_embed.weight)
#         # self.tok_embed.weight.requires_grad_(True)
#         # print('self.tok_embed.weight.requires_grad', self.tok_embed.weight.requires_grad)
#
#     def forward(self, x):
#         # print('x.device', x.device)
#         seq_len = x.size(1)  # x: [batch_size, seq_len]
#
#         pos = torch.arange(seq_len, device=device, dtype=torch.long)  # [seq_len]
#         # print('pos.device', pos.device)
#         # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
#         #         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
#
#         # expand_as类似于expand，只是目标规格是x.shape
#         pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
#
#         # 混合3种嵌入向量
#         # embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
#         # embedding = self.tok_embed(x) + self.pos_embed(pos)
#
#         embedding = self.pos_embed(pos)
#         embedding = embedding + self.tok_embed(x)
#
#         # layerNorm
#         embedding = self.norm(embedding)
#         return embedding
#
#
# # 计算Self-Attention
# class ScaledDotProductAttention(nn.Module):
#     def __init__(self):
#         super(ScaledDotProductAttention, self).__init__()
#
#     def forward(self, Q, K, V, attn_mask):
#         #  这里接收的Q, K, V是q_s, q_k, q_v，也就是真正的Q, K, V是向量，shape:[bach_size, seq_len, d_model]
#         # Q: [batch_size, n_head, seq_len, d_k]
#         # K: [batch_size, n_head, seq_len, d_k]
#         # V: [batch_size, n_head, seq_len, d_v]
#
#         scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_head, seq_len, seq_len]
#
#         # mask_filled:是将mask中为1/TRUE的元素所在的索引，在原tensor中相同的的索引处替换为指定的value
#         # remark: mask必须是一个 ByteTensor而且shape必须和a一样，mask value必须同为tensor
#         scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
#
#         attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_head, seq_len, seq_len]
#         context = torch.matmul(attn, V)  # [batch_size, n_head, seq_len, d_v]
#         return context, attn
#
#
# # 多头注意力机制
# class MultiHeadAttention(nn.Module):
#     def __init__(self):
#         super(MultiHeadAttention, self).__init__()
#         self.W_Q = nn.Linear(d_model, d_k * n_head)
#         self.W_K = nn.Linear(d_model, d_k * n_head)
#         self.W_V = nn.Linear(d_model, d_v * n_head)
#
#         self.linear = nn.Linear(n_head * d_v, d_model)
#         self.norm = nn.LayerNorm(d_model)
#
#     def forward(self, Q, K, V, attn_mask):
#         #  这里接收的Q, K, V都是enc_inputs，也就是embedding后的输入，shape:[bach_size, seq_len, d_model]
#         # Q: [batch_size, seq_len, d_model]
#         # K: [batch_size, seq_len, d_model]
#         # V: [batch_size, seq_len, d_model]
#         residual, batch_size = Q, Q.size(0)
#
#         # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
#         # 多头注意力是同时计算的，一次tensor乘法即可，这里是将多头注意力进行切分
#
#         # print('Q', Q.size(), Q)
#         # print('self.W_Q(Q)', self.W_Q(Q))
#         # print('self.W_Q(Q).view(batch_size, -1, n_head, d_k)', self.W_Q(Q).view(batch_size, -1, n_head, d_k))
#         # print('self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)',
#         #       self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2))
#
#         q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # q_s: [batch_size, n_head, seq_len, d_k]
#         k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # k_s: [batch_size, n_head, seq_len, d_k]
#         v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)  # v_s: [batch_size, n_head, seq_len, d_v]
#
#         # 处理前attn_mask: [batch_size, seq_len, seq_len]
#         attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
#         # 处理后attn_mask: [batch_size, n_head, seq_len, seq_len]
#
#         # context: [batch_size, n_head, seq_len, d_v], attn: [batch_size, n_head, seq_len, seq_len]
#         context, attention_map = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
#         context = context.transpose(1, 2).contiguous().view(batch_size, -1,
#                                                             n_head * d_v)  # context: [batch_size, seq_len, n_head * d_v]
#
#         output = self.linear(context)
#         output = self.norm(output + residual)
#         return output, attention_map
#
#
# # 基于位置的全连接层
# class PoswiseFeedForwardNet(nn.Module):
#     def __init__(self):
#         super(PoswiseFeedForwardNet, self).__init__()
#         self.fc1 = nn.Linear(d_model, d_ff)
#         self.fc2 = nn.Linear(d_ff, d_model)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
#         return self.fc2(self.relu(self.fc1(x)))
#
#
# class EncoderLayer(nn.Module):
#     def __init__(self):
#         super(EncoderLayer, self).__init__()
#         self.enc_self_attn = MultiHeadAttention()
#         self.pos_ffn = PoswiseFeedForwardNet()
#         self.attention_map = None
#
#     def forward(self, enc_inputs, enc_self_attn_mask):
#         # 多头注意力模块
#         enc_outputs, attention_map = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
#                                                         enc_self_attn_mask)  # enc_inputs to same Q,K,V
#         self.attention_map = attention_map
#         # 全连接模块
#         enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
#         return enc_outputs
#
#
# class Embeddings(nn.Module):
#     '''
#     对图像进行编码，把图片当做一个句子，把图片分割成块，每一块表示一个单词
#     '''
#
#     def __init__(self):
#         super(Embeddings, self).__init__()
#         img_size = 28
#         patch_size = 4
#         in_channels = 1
#         ##将图片分割成多少块（28/4）*（28/4）=49
#         n_patches = (img_size // patch_size) * (img_size // patch_size)
#         # 对图片进行卷积获取图片的块，并且将每一块映射成config.hidden_size维（32）
#         self.patch_embeddings = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=patch_size, stride=patch_size)
#
#         # 设置可学习的位置编码信息，（1,49+1,32）
#         self.position_embeddings = nn.Parameter(torch.zeros(1,
#                                                             n_patches + 1,
#                                                             32))
#         # 设置可学习的分类信息的维度
#         self.classifer_token = nn.Parameter(torch.zeros(1, 1, 32))
#         self.dropout = nn.Dropout(p=0.1)
#
#     def forward(self, x):
#         bs = x.shape[0]
#         cls_tokens = self.classifer_token.expand(bs, -1, -1) # (bs, 1, 32)
#         x = self.patch_embeddings(x)  # （bs,32,7,7）
#         x = x.flatten(2)  # (bs,32,49)
#         x = x.transpose(-1, -2)  # (bs,49,32)
#         x = torch.cat((cls_tokens, x), dim=1)  # 将分类信息与图片块进行拼接（bs,50,32）
#         embeddings = x + self.position_embeddings  # 将图片块信息和对其位置信息进行相加(bs,50,32)
#         embeddings = self.dropout(embeddings)
#         print(embeddings)
#         return embeddings
#
#
# # 完整的模型
# class Transformer_Encoder(nn.Module):
#     def __init__(self, config):
#         super(Transformer_Encoder, self).__init__()
#         self.config = config
#         self.input_size = [1, 28, 28]
#
#         global max_len, n_layers, n_head, d_model, d_ff, d_k, d_v, vocab_size, device
#         # max_len = config.max_len + 2
#         n_layers = config.num_layer
#         n_head = config.num_head
#         d_model = config.dim_embedding
#         d_ff = config.dim_feedforward
#         d_k = config.dim_k
#         d_v = config.dim_v
#         # vocab_size = config.vocab_size
#         device = torch.device("cuda" if config.cuda else "cpu")
#
#         # print(max_len)
#         # print(vocab_size)
#         # print(d_model)
#         # exit(0)
#
#         # Embedding Layer
#         self.embedding = Embeddings()
#
#         # Encoder Layer
#         self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])  # 定义重复的模块
#
#         # Task-specific Layer
#         self.fc_task = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.Dropout(0.5),
#             nn.ReLU(),
#             nn.Linear(d_model, d_model),
#         )
#
#         if config.mode == 'train-test' or config.mode == 'cross validation':
#             ways = config.num_class
#             self.classifier = nn.Linear(d_model, ways)
#
#     def forward(self, input_ids):
#         input_ids = input_ids.float()
#         x = input_ids.view(len(input_ids), 1, self.input_size[1], self.input_size[2])
#         print("input_ids", x.size())
#         # embedding layer
#         output = self.embedding(x)  # [bach_size, seq_len, d_model]
#         print(output.size())
#
#         # print('view x', input_ids.size())
#         # output = input_ids.view(len(input_ids), self.input_size[1], self.input_size[2])
#         # print('view x', output.size())
#
#         # 获取掩模判断矩阵
#         enc_self_attn_mask = get_attn_pad_mask(input_ids)  # [batch_size, maxlen, maxlen]
#         print("enc_self_attn_mask", enc_self_attn_mask.size())
#
#         # encoder layer
#         for layer in self.layers:
#             output = layer(output, enc_self_attn_mask)
#             # output: [batch_size, max_len, d_model]
#
#         # task-specific layer
#         # [CLS] only needed
#         output = output[:, 0, :]
#         embeddings = self.fc_task(output)
#         print("embeddings", embeddings.size())
#         embeddings = embeddings.view(embeddings.size(0), -1)
#         print("embeddings", embeddings.size())
#
#         if self.config.mode == 'train-test' or self.config.mode == 'cross validation':
#             logits_clsf = self.classifier(embeddings)
#         else:
#             logits_clsf = None
#
#         return logits_clsf, embeddings
#
#
# def check_model():
#     config = config_meta.get_config()
#     torch.cuda.set_device(config.device)  # 选择要使用的GPU
#
#     # 加载词典
#     residue2idx = pickle.load(open(config.path_token2index + 'residue2idx.pkl', 'rb'))
#     config.vocab_size = len(residue2idx)
#
#     model = Transformer_Encoder(config)
#
#     print('-' * 50, 'Model', '-' * 50)
#     print(model)
#
#     print('-' * 50, 'Model.named_parameters', '-' * 50)
#     for name, value in model.named_parameters():
#         print('[{}]->[{}],[requires_grad:{}]'.format(name, value.shape, value.requires_grad))
#
#     # 冻结网络的部分层
#     # util_freeze.freeze_by_names(model, ['embedding', 'layers'])
#     util_model.freeze_by_idxs(model, [0, 1])
#
#     print('-' * 50, 'Model.named_children', '-' * 50)
#     for name, child in model.named_children():
#         print('\\' * 40, '[name:{}]'.format(name), '\\' * 40)
#         print('child:\n{}'.format(child))
#
#         if name == 'soft_attention':
#             print('soft_attention')
#             for param in child.parameters():
#                 print('param.shape', param.shape)
#                 print('param.requires_grad', param.requires_grad)
#
#         for sub_name, sub_child in child.named_children():
#             print('*' * 20, '[sub_name:{}]'.format(sub_name), '*' * 20)
#             print('sub_child:\n{}'.format(sub_child))
#
#             # if name == 'layers' and (sub_name == '5' or sub_name == '4'):
#             if name == 'layers' and (sub_name == '5'):
#                 print('Ecoder 5 is unfrozen')
#                 for param in sub_child.parameters():
#                     param.requires_grad = True
#
#         # for param in child.parameters():
#         #     print('param.requires_grad', param.requires_grad)
#
#     print('-' * 50, 'Model.named_parameters', '-' * 50)
#     for name, value in model.named_parameters():
#         print('[{}]->[{}],[requires_grad:{}]'.format(name, value.shape, value.requires_grad))
#
#
# def forward_test():
#     config = config_meta.get_config()
#     torch.cuda.set_device(config.device)  # 选择要使用的GPU
#
#     # 加载词典
#     residue2idx = pickle.load(open(config.path_token2index + 'residue2idx.pkl', 'rb'))
#     config.vocab_size = len(residue2idx)
#
#     model = Transformer_Encoder(config)
#
#     input = torch.randint(28, [4, 20])
#
#     if config.cuda:
#         device = torch.device('cuda')
#         model = model.to(device)
#         input = input.to(device)
#
#     output = model(input)
#     print('output', output)
#
#
# if __name__ == '__main__':
#     # check model
#     check_model()
#     # forward test
#     forward_test()

#1.编码
class Embeddings(nn.Module):
    '''
    对图像进行编码，把图片当做一个句子，把图片分割成块，每一块表示一个单词
    '''

    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.config = config
        img_size = config.img_size
        patch_size = config.patch_size
        in_channels = config.in_channels
        ##将图片分割成多少块（28/4）*（28/4）=49
        n_patches = (img_size // patch_size) * (img_size // patch_size)
        # 对图片进行卷积获取图片的块，并且将每一块映射成config.hidden_size维（32）
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels, out_channels=config.hidden_size, kernel_size=patch_size, stride=patch_size)

        # 设置可学习的位置编码信息，（1,49+1,32）
        self.position_embeddings = nn.Parameter(torch.zeros(1,
                                                            n_patches + 1,
                                                            32))
        # 设置可学习的分类信息的维度
        self.classifer_token = nn.Parameter(torch.zeros(1, 1, 32))
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        bs = x.shape[0]
        cls_tokens = self.classifer_token.expand(bs, -1, -1) # (bs, 1, 32)
        x = self.patch_embeddings(x)  # （bs,32,7,7）
        x = x.flatten(2)  # (bs,32,49)
        x = x.transpose(-1, -2)  # (bs,49,32)
        x = torch.cat((cls_tokens, x), dim=1)  # 将分类信息与图片块进行拼接（bs,50,32）
        embeddings = x + self.position_embeddings  # 将图片块信息和对其位置信息进行相加(bs,50,32)
        embeddings = self.dropout(embeddings)
        # print(embeddings.size())
        return embeddings

#2.构建self-Attention模块
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.config = config
        self.vis = vis
        self.num_attention_heads = config.num_head
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  # 32/8=4
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 4*8=32

        self.query = nn.Linear(config.hidden_size, self.all_head_size)#wm,32->32，Wq矩阵为（32,32）
        self.key = nn.Linear(config.hidden_size, self.all_head_size)#wm,32->32,Wk矩阵为（32,32）
        self.value = nn.Linear(config.hidden_size, self.all_head_size)#wm,32->32,Wv矩阵为（32,32）
        self.out = nn.Linear(config.hidden_size, config.hidden_size)  # wm,32->32
        self.attn_dropout = nn.Dropout(p=0.0)
        self.proj_dropout = nn.Dropout(p=0.0)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
        self.num_attention_heads, self.attention_head_size)  # wm,(bs,50)+(8,4)=(bs,50,8,4)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # wm,(bs,8,50,4)

    def forward(self, hidden_states):
        # hidden_states为：(bs,50,32)
        mixed_query_layer = self.query(hidden_states)#wm,32->32
        mixed_key_layer = self.key(hidden_states)#wm,32->32
        mixed_value_layer = self.value(hidden_states)#wm,32->32

        query_layer = self.transpose_for_scores(mixed_query_layer)#wm，(bs,8,50,4)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))#将q向量和k向量进行相乘（bs,4,50,50)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)#将结果除以向量维数的开方
        attention_probs = self.softmax(attention_scores)#将得到的分数进行softmax,得到概率
        weights = attention_probs if self.vis else None#wm,实际上就是权重
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)#将概率与内容向量相乘
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)#wm,(bs,50)+(32,)=(bs,50,32)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights#wm,(bs,50,32),(bs,50,50)

#3.构建前向传播神经网络
#两个全连接神经网络，中间加了激活函数
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, 256)#wm,32->256
        self.fc2 = nn.Linear(256, config.hidden_size)#wm,256->32
        self.act_fn = torch.nn.functional.gelu#wm,激活函数
        self.dropout = nn.Dropout(p=0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)#wm,32->256
        x = self.act_fn(x)#激活函数
        x = self.dropout(x)#wm,丢弃
        x = self.fc2(x)#256->32
        x = self.dropout(x)
        return x


# 4.构建编码器的可重复利用的Block()模块：每一个block包含了self-attention模块和MLP模块
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size  # wm,32
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)  # wm，层归一化
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h  # 残差结构

        hh = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + hh  # 残差结构
        return x, weights

#5.构建Encoder模块，该模块实际上就是堆叠N个Block模块
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.num_layer):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

#6构建transformers完整结构，首先图片被embedding模块编码成序列数据，然后送入Encoder中进行编码
class Transformer(nn.Module):
    def __init__(self, config, vis):
        super(Transformer, self).__init__()
        self.img_size = config.img_size
        self.embeddings = Embeddings(config)#wm,对一幅图片进行切块编码，得到的是（bs,n_patch+1（50）,每一块的维度（32））
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)#wm,输出的是（bs,50,32)
        encoded, attn_weights = self.encoder(embedding_output)#wm,输入的是（bs,50,32)
        return encoded, attn_weights #输出的是（bs,50,32）

#7构建VisionTransformer，用于图像分类
class VisionTransformer(nn.Module):
    def __init__(self, config):
        super(VisionTransformer, self).__init__()
        self.zero_head = False
        self.vis = False
        self.config = config
        self.transformer = Transformer(config, self.vis)

        # Task-specific Layer
        self.fc_task = nn.Sequential(
            nn.Linear(32, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        if config.mode == 'train-test' or config.mode == 'cross validation':
            ways = config.num_class
            self.classifier = nn.Linear(config.hidden_size, ways)  # wm,32-->2

    def forward(self, x):
        x = x.float()
        x = x.view(len(x), 1, self.config.img_size, self.config.img_size)
        # print("input_ids", x.size())
        x, attn_weights = self.transformer(x)
        # print("input_ids", x.size())
        output = x[:, 0, :]
        embeddings = self.fc_task(output)
        # print("embeddings", embeddings.size())
        embeddings = embeddings.view(embeddings.size(0), -1)
        # print("embeddings", embeddings.size())

        #如果传入真实标签，就直接计算损失值
        # if labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        #     return loss
        # else:
        #     return logits, attn_weights
        if self.config.mode == 'train-test' or self.config.mode == 'cross validation':
            logits_clsf = self.classifier(embeddings)
        else:
            logits_clsf = None

        return logits_clsf, embeddings