
from torch.nn import Module, LayerNorm, ELU, Identity, Linear, Sequential
from torch.nn.functional import relu


# Multi-layer Perceptron
class MLP(Module):

    def __init__(self,
                 input_size,
                 layer_sizes,
                 output_size,
                 output_activation=Identity,
                 activation=ELU):
        super().__init__()
        sizes = [input_size] + layer_sizes + [output_size]
        layers = []
        for i in range(len(sizes) - 1):
            act = activation if i < len(sizes) - 2 else output_activation
            layers += [Linear(sizes[i], sizes[i + 1]), act()]
        self.network = Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Residual block
class ResidualBlock(Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.ln1 = LayerNorm(input_size)
        self.mlp1 = MLP(input_size, 2 * [hidden_size], input_size)
        self.ln2 = LayerNorm(input_size)
        self.mlp2 = MLP(input_size, 2 * [hidden_size], input_size)

    def forward(self, x):
        out = self.ln1(x)
        out = relu(out)
        out = self.mlp1(out)
        out = self.ln2(out)
        out = relu(out)
        out = self.mlp2(out)
        out += x
        return out