import torch
import torch.nn as nn


class BernoulliEmitter(nn.Module):
    """
    Parameterizes the bernoulli observation likelihood `p(x_t | z_t)`
    """

    def __init__(self, input_dim, z_dim, emission_dim):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)

        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()

    def forward(self, z_t):
        """
        Given the latent z at a particular time step t we return the vector of
        probabilities `ps` that parameterizes the bernoulli distribution `p(x_t|z_t)`
        """
        h1 = self.relu(self.lin_z_to_hidden(z_t))  # emission_dim
        h2 = self.relu(self.lin_hidden_to_hidden(h1))  # emission_dim
        ps = torch.sigmoid(self.lin_hidden_to_input(h2))  # input_dim

        return ps


class NormalEmitter(nn.Module):
    """
    Parameterizes the bernoulli observation likelihood `p(x_t | z_t)`
    """

    def __init__(self, input_dim, z_dim, emission_dim):
        super().__init__()
        self.lin_z_to_gate_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_gate_hidden_to_input = nn.Linear(emission_dim, input_dim)

        self.lin_z_to_mean_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_mean_hidden_to_input = nn.Linear(emission_dim, input_dim)

        self.lin_z_to_loc = nn.Linear(z_dim, input_dim)

        self.lin_z_to_scale = nn.Linear(z_dim, input_dim)

        # non lineaities
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.soft_plus = nn.Softplus()

    def forward(self, z_t):
        """
        Given the latent z at a particular time step t we return the vector of
        probabilities `ps` that parameterizes the bernoulli distribution `p(x_t|z_t)`
        """
        _gate = self.relu(self.lin_z_to_gate_hidden(z_t))
        gate = self.sigmoid(self.lin_gate_hidden_to_input(_gate))

        _mean = self.relu(self.lin_z_to_mean_hidden(z_t))
        mean = self.lin_mean_hidden_to_input(_mean)

        _loc = self.lin_z_to_loc(z_t)
        loc = (1 - gate) * _loc + gate * mean

        scale = self.soft_plus(self.lin_z_to_scale(z_t))

        return loc, scale


