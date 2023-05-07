import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        return lr


class Linear_Mapping(nn.Module):
    def __init__(self, m):
        super(Linear_Mapping, self).__init__()
        self.m = m
        self.f = nn.Sequential(
            nn.Linear(2 * m, 1),
        )

    def forward(self, input, detach=True):
        if detach:
            input = input.detach()
        a = self.f(input.transpose(1, 2)).transpose(1, 2)
        a = a / torch.sqrt(torch.sum(a**2, dim=2, keepdim=True))

        return a


class Non_Linear_Mapping(nn.Module):
    def __init__(self, m, d):
        super(Non_Linear_Mapping, self).__init__()
        self.m = m
        self.d = d
        self.f = nn.Sequential(
            nn.Linear(2 * m, 1),
        )
        self.h = nn.Sequential(
            nn.Linear(d, d),
            nn.Sigmoid(),
            nn.Linear(d, d),
        )

    def forward(self, input, detach=True):
        if detach:
            input = input.detach()
        a = self.f(input.transpose(1, 2)).transpose(1, 2)
        a = self.h(a)
        a = a / torch.sqrt(torch.sum(a**2, dim=2, keepdim=True))

        return a


class Generalized_Linear_Mapping(nn.Module):
    def __init__(self, m, d):
        super(Generalized_Linear_Mapping, self).__init__()
        self.m = m
        self.d = d
        self.f = nn.Sequential(
            nn.Linear(2 * m, 1),
        )
        self.g = nn.Sequential(
            nn.Linear(d, d),
            nn.Sigmoid(),
            nn.Linear(d, d),
        )

    def forward(self, input, detach=True):
        if detach:
            input = input.detach()
        a = self.f(self.g(input).transpose(1, 2)).transpose(1, 2)
        a = a / torch.sqrt(torch.sum(a**2, dim=2, keepdim=True))

        return a


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q / self.temperature, k.transpose(1, 2))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.bmm(attn, v)

        return output


class Attention(nn.Module):
    """Single-Head Attention module"""

    def __init__(self, d=3, d_k=64, d_v=3, dropout=0):
        super(Attention, self).__init__()

        self.toqueries = nn.Linear(d, d_k)
        self.tokeys = nn.Linear(d, d_k)
        self.tovalues = nn.Linear(d, d_v)
        self.attn = ScaledDotProductAttention(temperature=d_k**0.5, attn_dropout=dropout)

    def forward(self, input, mask=None):
        # Pass through the pre-attention projection:
        # input: batch_size x m x d
        q = self.toqueries(input)
        k = self.tokeys(input)
        v = self.tovalues(input)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q = self.attn(q, k, v, mask=mask)

        return q.mean(dim=1, keepdim=True)


class EfficientAttention(nn.Module):
    """
    https://github.com/cmsflash/efficient-attention
    """

    def __init__(self, d=3, d_k=64, d_v=3, dropout=0):
        super(EfficientAttention, self).__init__()

        self.toqueries = nn.Linear(d, d_k)
        self.tokeys = nn.Linear(d, d_k)
        self.tovalues = nn.Linear(d, d_v)

    def forward(self, input, mask=None):
        # Pass through the pre-attention projection:
        # input: batch_size x m x d
        q = self.toqueries(input)
        k = self.tokeys(input)
        v = self.tovalues(input)

        # batch_size x m x d_k
        q = F.softmax(q, dim=2)
        # batch_size x m x d_k
        k = F.softmax(k, dim=1)
        # batch_size x d_k x d_v
        context = torch.bmm(k.transpose(1, 2), v)
        # batch_size x m x d_v
        q = torch.bmm(q, context)

        return q.mean(dim=1, keepdim=True)


def get_proj(input_size, output_size):
    lin = nn.Linear(input_size, output_size)
    torch.nn.init.xavier_normal_(lin.weight)
    return lin


class LinearAttention(nn.Module):
    """
    https://github.com/tatp22/linformer-pytorch
    """

    def __init__(self, m, l, d=3, d_k=64, d_v=3, sharing=False, dropout=0):
        super(LinearAttention, self).__init__()

        self.toqueries = nn.Linear(d, d_k)
        self.tokeys = nn.Linear(d, d_k)
        self.tovalues = nn.Linear(d, d_v)
        if sharing:
            self.Kproj = self.Vproj = get_proj(m, l)
        else:
            self.Kproj = get_proj(m, l)
            self.Vproj = get_proj(m, l)
        self.attn = ScaledDotProductAttention(temperature=d_k**0.5, attn_dropout=dropout)

    def forward(self, input, mask=None):
        # Pass through the pre-attention projection:
        # input: batch_size x m x d
        q = self.toqueries(input)
        k = self.tokeys(input)
        v = self.tovalues(input)

        # k_proj = batch_size x l x d_k
        k_proj = self.Kproj(k.transpose(1, 2)).transpose(1, 2)
        # v_proj = batch_size x l x d_v
        v_proj = self.Vproj(v.transpose(1, 2)).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q = self.attn(q, k_proj, v_proj, mask=mask)

        return q.mean(dim=1, keepdim=True)


class Attention_Mapping(nn.Module):
    def __init__(self, m, d, d_k, dropout=0):
        super(Attention_Mapping, self).__init__()
        self.attn = Attention(d=d, d_k=d_k, d_v=d, dropout=dropout)

    def forward(self, input, detach=True):
        if detach:
            input = input.detach()
        x, y = torch.chunk(input, 2, dim=1)
        assert x.size() == y.size()
        x = self.attn(x)
        y = self.attn(y)
        a = (x + y) / 2
        a = a / torch.sqrt(torch.sum(a**2, dim=2, keepdim=True))

        return a


class EfficientAttention_Mapping(nn.Module):
    def __init__(self, m, d, d_k, dropout=0):
        super(EfficientAttention_Mapping, self).__init__()
        self.attn = EfficientAttention(d=d, d_k=d_k, d_v=d, dropout=dropout)

    def forward(self, input, detach=True):
        if detach:
            input = input.detach()
        x, y = torch.chunk(input, 2, dim=1)
        assert x.size() == y.size()
        x = self.attn(x)
        y = self.attn(y)
        a = (x + y) / 2
        a = a / torch.sqrt(torch.sum(a**2, dim=2, keepdim=True))

        return a


class LinearAttention_Mapping(nn.Module):
    def __init__(self, m, l, d, d_k, sharing=False, dropout=0):
        super(LinearAttention_Mapping, self).__init__()
        self.attn = LinearAttention(m=m, l=l, d=d, d_k=d_k, d_v=d, sharing=sharing, dropout=dropout)

    def forward(self, input, detach=True):
        if detach:
            input = input.detach()
        x, y = torch.chunk(input, 2, dim=1)
        assert x.size() == y.size()
        x = self.attn(x)
        y = self.attn(y)
        a = (x + y) / 2
        a = a / torch.sqrt(torch.sum(a**2, dim=2, keepdim=True))

        return a
