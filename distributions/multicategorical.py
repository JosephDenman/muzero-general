import torch
from torch.distributions import Categorical, Distribution


class MultiCategorical(Distribution):

    def __init__(self, logits_batch):
        """
        :param logits: A two-dimensional tensor. The first dimension represents parameters
                       for each component distribution. The second dimension represents the
                       class probabilities for a component distribution.
        """
        super().__init__()
        for logits in logits_batch:
            print(logits)
        self.dists = [Categorical(logits) for logits in logits_batch]

    def log_prob(self, value):
        ans = []
        for d, v in zip(self.dists, torch.split(value, 1, dim=-1)):
            ans.append(d.log_prob(v.squeeze(-1)))
        return torch.stack(ans, dim=-1).sum(dim=-1)

    def entropy(self):
        return torch.stack([d.entropy() for d in self.dists], dim=-1).sum(dim=-1)

    def sample(self, sample_shape=torch.Size()):
        return torch.stack([d.sample(sample_shape) for d in self.dists], dim=-1)

    def sample_n(self, n):
        return torch.stack([self.sample() for _ in range(n)], dim=0)
