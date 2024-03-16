from torch.autograd import Function
import torch
import sys


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def bpr_loss(pos, neg):
    loss = -torch.mean(torch.log2(torch.sigmoid(pos - neg)))
    return loss


def causal_loss(s_input, t_input, s_causal, t_causal, adj, tag):
    mse_loss = torch.nn.MSELoss()
    if tag:
        rec_loss = 0 * mse_loss(s_input, s_causal) + mse_loss(t_input, t_causal)
    else:
        rec_loss = mse_loss(s_input, s_causal) + mse_loss(t_input, t_causal)
    dim = adj.shape[0]
    dag_loss = torch.tensor(float(dim))
    factorial = 1.0
    p_power = torch.eye(dim)
    p = adj * adj

    for i in range(1, dim):
        p_power = torch.mm(p_power, p)
        factorial = factorial * float(i)
        dag_loss = dag_loss + p_power.trace() * (1.0 / factorial)

    dag_loss = dag_loss - torch.tensor(float(dim))

    spa_loss = torch.norm(adj, p=1)

    i2u_loss = torch.norm(adj[-dim // 2:, 0:dim // 2], p=1)

    notr_loss = torch.tensor(0.)
    for col in range(dim // 2, dim):
        notr_loss = notr_loss + torch.log2(torch.norm(adj[:, col], p=1))
    notr_loss = notr_loss / (dim * dim / 2)

    return rec_loss + dag_loss * dag_loss + 6.0 * dag_loss + 1e-5 * spa_loss + 1e-5 * i2u_loss - notr_loss
