from abc import ABC, abstractmethod
from torch import Tensor, tanh, relu, split
from torch.nn import Module, Linear, LayerNorm, Tanh, DataParallel
from common import MLP, ResidualBlock
from distributions.multicategorical import MultiCategorical
import torch


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class AbstractNetwork(ABC, Module):

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
    def sample_actions(self):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


class ContinuousControlNetwork(AbstractNetwork):
    """
    Used for one-dimensional continuous control tasks.
    """

    def __init__(self,
                 observation_size,
                 state_size,
                 action_size,
                 sample_size,
                 rn_ll_output_size,
                 rn_fc_hidden_size,
                 rn_rt_size,
                 dn_fc_hidden_size,
                 dn_rt_size,
                 pn_fc_hidden_size,
                 bin_size,
                 vn_fc_hidden_size,
                 full_support_size,
                 reference_distribution=None):
        super().__init__()
        self.sample_size = sample_size
        self.action_size = action_size
        self.bin_size = bin_size
        self.reference_distribution = reference_distribution
        self.policy_distribution = None
        self.representation_network = DataParallel(
            RepresentationNetwork(
                observation_size,
                rn_ll_output_size,
                rn_fc_hidden_size,
                rn_rt_size
            )
        )
        self.dynamics_network = DataParallel(
            DynamicsNetwork(
                state_size,
                action_size,
                dn_fc_hidden_size,
                dn_rt_size
            )
        )
        self.prediction_network = DataParallel(
            PredictionNetwork(
                state_size,
                action_size,
                pn_fc_hidden_size,
                bin_size,
                vn_fc_hidden_size,
                full_support_size
            )
        )
        self.reward_network = RewardNetwork()

    # TODO: Test this
    def initial_inference(self, observation, legal_actions):
        encoded_state = self.representation(observation)
        reward = 0  # TODO: Reshape reward appropriately
        value, sampled_actions, policy_logits, empirical_logits, reference_logits = self.prediction(encoded_state)
        return value, reward, sampled_actions, policy_logits, empirical_logits, reference_logits

    def recurrent_inference(self, encoded_state, action):
        """
        :returns: Value, reward, sampled_actions, and the policy, empirical, and reference probabilities
                  (as dictionaries), and encoded state.
        """
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        value, sampled_actions, policy_logits, empirical_logits, reference_logits = self.prediction(
            next_encoded_state)
        return value, reward, sampled_actions, policy_logits, empirical_logits, reference_logits

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
                .min(2, keepdim=True)[0]
                .unsqueeze(-1)
        )
        max_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
                .max(2, keepdim=True)[0]
                .unsqueeze(-1)
        )
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (encoded_state - min_encoded_state) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        encoded_state = self.dynamics_network(encoded_state, action)
        reward = self.reward_network(encoded_state)
        return encoded_state, reward

    def prediction(self, encoded_state):
        policy_params, value = self.prediction_network(encoded_state)
        self.policy_distribution = MultiCategorical(split(policy_params, self.bin_size))
        if self.reference_distribution is None:
            self.reference_distribution = self.policy_distribution
        sampled_actions = self.sample_actions()
        policy_logits = self.policy_logits(sampled_actions)
        empirical_logits = self.empirical_logits(sampled_actions)
        reference_logits = self.reference_logits(sampled_actions)
        return value, sampled_actions, policy_logits, empirical_logits, reference_logits

    def sample_actions(self, legal_actions=None):
        if legal_actions is None:
            return self.reference_distribution.sample_n(self.sample_size)
        else:
            legal_samples = []
            for _ in range(self.sample_size):
                sample = self.reference_distribution.sample_actions()
                if sample in legal_actions:
                    legal_samples.append(sample)
            return legal_samples

    # TODO: Should these return log probabilities? May have to compensate in loss function, and in improvement operator.
    def policy_logits(self, actions):
        """
        :returns: A dictionary mapping the given actions to their
                  probabilities according to the current policy network.
        """
        return {a: self.policy_distribution.log_prob(a) for a in actions}

    def reference_logits(self, actions):
        """
        :returns: A dictionary mapping the given actions to their
                  probabilities according to the reference policy.
        """
        return {a: self.reference_distribution.log_prob(a) for a in actions}

    @staticmethod
    def empirical_logits(actions):
        empirical_logits = {}
        for a in actions:
            empirical_logits[a] = empirical_logits.get(a, 0) + 1
        return empirical_logits


class RepresentationNetwork(Module):

    def __init__(self,
                 observation_size,
                 ll_output_size,
                 fc_hidden_size,
                 residual_tower_size):
        super().__init__()
        self.ll = Linear(observation_size, ll_output_size)
        self.ln = LayerNorm(ll_output_size)
        self.rbs = [ResidualBlock(ll_output_size,
                                  fc_hidden_size) for _ in range(residual_tower_size)]

    def forward(self, x):
        out = self.ll(x)
        out = self.ln(out)
        out = tanh(out)
        for residual_block in self.rbs:
            out = residual_block(out)
        return out


class DynamicsNetwork(Module):

    def __init__(self,
                 state_size,
                 action_size,
                 fc_hidden_size,
                 residual_tower_size):
        super().__init__()
        self.state_size = state_size
        self.ll1 = Linear(action_size, state_size)
        self.ln1 = LayerNorm(state_size)
        self.ll2 = Linear(state_size, state_size)
        self.ln2 = LayerNorm(state_size)
        self.rbs = [ResidualBlock(state_size,
                                  fc_hidden_size) for _ in range(residual_tower_size)]

    def forward(self, state, action):
        out = self.ll1(action)
        out = self.ln1(out)
        out = relu(out)
        out = self.ll2(out + state)
        out = self.ln2(out)
        out = tanh(out)
        for residual_block in self.rbs:
            out = residual_block(out)
        return out


class PredictionNetwork(Module):

    def __init__(self,
                 state_size,
                 action_size,
                 pn_fc_hidden_size,
                 bin_size,
                 vn_fc_hidden_size,
                 full_support_size):
        super().__init__()
        self.policy_mlp = MLP(state_size,
                              2 * [pn_fc_hidden_size],
                              action_size * bin_size,
                              activation=Tanh)
        self.value_mlp = MLP(state_size,
                             2 * [vn_fc_hidden_size],
                             full_support_size)

    def forward(self, x):
        return self.policy_mlp(x), self.value_mlp(x)
