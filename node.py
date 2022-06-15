import numpy


def normalize(logits):
    """
    :param logits: A dictionary mapping actions to logits
    :return:       A dictionary representing a normalized probability distribution
    """
    normalization_constant = 1.0 / sum(logits.values())
    for action, logit in logits.items():
        logits[action] = logit * normalization_constant
    return logits


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, sampled_actions, to_play, reward, policy_logits, empirical_logits, reference_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        :param sampled_actions:  A list of actions sampled from the reference distribution
        :param policy_logits:    A dictionary mapping actions to policy logits
        :param empirical_logits: A dictionary mapping actions to empirical logits
        :param reference_logits: A dictionary mapping actions to reference logits
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state
        policy = normalize({a: empirical_logits[a] / reference_logits[a] * policy_logits[a] for a in sampled_actions})
        for action, p in policy.items():
            self.children[action] = Node(p)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac