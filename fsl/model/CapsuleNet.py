import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class CapsuleNet(nn.Module):
    """
    A Capsule Network.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """

    def __init__(self, config):
        super(CapsuleNet, self).__init__()
        self.config = config
        self.input_size = [1, 28, 28]

        conv_channels_in = config.conv_channels_in
        conv_channels_out = config.conv_channels_out
        kernel_sizes = [int(kss) for kss in config.kernel_sizes.split(',')]
        strides = [int(sts) for sts in config.strides.split(',')]

        dim_cap_in = config.dim_cap_in
        dim_cap_out = config.dim_cap_out
        out_num_caps = config.num_caps_out
        in_num_caps = config.num_caps_in

        routings = config.routings

        # vocab_size = config.vocab_size
        # embedding_dim = config.dim_embedding
        #
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # if config.static:  # 如果使用预训练词向量，则提前加载，当不需要微调时设置freeze为True
        #     self.embedding = self.embedding.from_pretrained(config.vectors, freeze=not config.fine_tune)

        # Layer 1: Just a conventional Conv2D layer
        # self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=conv_channels_out, kernel_size=kernel_sizes[0], stride=strides[0], padding=0)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        # (in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0)
        self.primarycaps = PrimaryCapsule(in_channels=conv_channels_in, out_channels=conv_channels_out,
                                                       dim_caps=dim_cap_in, kernel_size=kernel_sizes[1],
                                                       stride=strides[1], padding=0)

        # Layer 3: Capsule layer. Routing algorithm works here.
        # (in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3)
        self.digitcaps = DenseCapsule(in_num_caps=in_num_caps, in_dim_caps=dim_cap_in,
                                      out_num_caps=out_num_caps, out_dim_caps=dim_cap_out, routings=routings)

        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(16 * out_num_caps, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.input_size[0] * self.input_size[1] * self.input_size[2]),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

        # if self.config.mode == 'train-test' or self.config.mode == 'cross validation':
        #     label_num = config.num_class
        #     self.classifier = nn.Linear(dim_cap_out, label_num)

    def forward(self, x, y=None):
        # x = self.embedding(x)
        # print(x.size())
        # x = x.view(x.size(0), 1, x.size(1), self.config.dim_embedding)
        x = x.float()
        x = x.view(x.size(0), 1, self.input_size[1], self.input_size[2])
        x = self.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        length = x.norm(dim=-1)

        if y is None:  # during testing, no label given. create one-hot coding using `length`
            index = length.max(dim=1)[1]
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).cuda())

        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        if self.config.mode == 'train-test' or self.config.mode == 'cross validation':
                # length = self.classifier(length)
                reconstruction = reconstruction.view(-1, *self.input_size)

        return length, reconstruction

def squash(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-8)
    return scale * inputs

class PrimaryCapsule(nn.Module):
    """
    Apply Conv2D with `out_channels` and then reshape to get capsules
    :param in_channels: input channels
    :param out_channels: output channels
    :param dim_caps: dimension of capsule
    :param kernel_size: kernel size
    :return: output tensor, size=[batch, num_caps, dim_caps]
    """
    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0):
        super(PrimaryCapsule, self).__init__()
        self.dim_caps = dim_caps
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        outputs = self.conv2d(x)
        outputs = outputs.view(x.size(0), -1, self.dim_caps)
        return squash(outputs)

class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.

    :param in_num_caps: number of cpasules inputted to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))

    def forward(self, x):
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)
        x_hat_detached = x_hat.detach()

        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps]
        b = torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)

        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            # c.size = [batch, out_num_caps, in_num_caps]
            c = F.softmax(b, dim=1)

            # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.routings - 1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat))  # alternative way
            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                device = torch.device('cuda:0')
                outputs = squash(torch.sum(c[:, :, :, None].to(device) * x_hat_detached.to(device), dim=-2, keepdim=True).to(device))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # alternative way

                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                b = b.to(device) + torch.sum(outputs.to(device) * x_hat_detached.to(device), dim=-1).to(device)

        return torch.squeeze(outputs, dim=-2)

