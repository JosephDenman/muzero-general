from abc import ABC, abstractmethod


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class AbstractNetwork(ABC, torch.nn.Module):

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation, legal_actions):
        """
        :returns: Value, reward, sampled_actions, and the policy, empirical, and reference probabilities
                  (as dictionaries), and encoded state.
        """
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        """
        :returns: Value, reward, sampled_actions, and the policy, empirical, and reference probabilities
                  (as dictionaries), and encoded state.
        """
        pass

    @abstractmethod
    def policy_logits(self, actions):
        """
        :returns: A dictionary mapping the given actions to their probabilities according to the current policy.
        """
        pass

    @abstractmethod
    def sample(self):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


class CCNetwork(AbstractNetwork):
    """
    Use for one-dimensional continuous control tasks.
    """

    def __init__(self, reference_distribution=None):
        super().__init__()
        self.representation_network = None
        self.dynamics_network = None
        self.policy_network = None # Do these two share weights?
        self.value_network = None
        if reference_distribution is None:
            self.reference_distribution = self.policy_network

    def initial_inference(self, observation, legal_actions):
        legal_samples = []
        if legal_actions is None:
            legal_samples = self.reference_distribution.sample_n(self.config.sample_size)
        else:
            remaining_samples = self.config.sample_size
            while remaining_samples > 0:
                new_sample = self.reference_distribution.sample(1)
                if new_sample in legal_actions:
                    remaining_samples -= 1
                    legal_samples.append(new_sample)
        # Evaluate policy network on each sample
        pass

    def recurrent_inference(self, encoded_state, action):
        """
        :returns: Value, reward, sampled_actions, and the policy, empirical, and reference probabilities
                  (as dictionaries), and encoded state.
        """
        pass

    def policy_logits(self, actions):
        """
        :returns: A dictionary mapping the given actions to their probabilities according to the current policy.
        """
        pass

    def sample(self):
        return self.reference_distribution.sample_n(self.config.sample_size)


class PolicyNetwork():
    def sample_n(self):
        pass


class ValueNetwork():
    pass


class RepresentationNetwork():
    pass


class DynamicsNetwork():
    pass
