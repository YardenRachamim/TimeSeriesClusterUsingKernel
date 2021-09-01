import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.distributions.transforms import affine_autoregressive
import pyro.contrib.examples.polyphonic_data_loader as poly
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution

from StohasticModels.Modules.Emitter import BernoulliEmitter, NormalEmitter
from StohasticModels.Modules.GatedTransition import NormalGatedTransition
from StohasticModels.Modules.Combiner import NormalCombiner


class DMM(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    """

    def __init__(
            self,
            input_dim=88,
            z_dim=100,
            emission_dim=100,
            transition_dim=200,
            rnn_dim=600,
            num_layers=1,
            rnn_dropout_rate=0.0,
            num_iafs=0,
            iaf_dim=50,
            use_cuda=False,
    ):
        super().__init__()
        # instantiate PyTorch modules used in the model and guide below
        self.emitter = NormalEmitter(input_dim, z_dim, emission_dim)
        self.trans = NormalGatedTransition(z_dim, transition_dim)
        self.combiner = NormalCombiner(z_dim, rnn_dim)
        # dropout just takes effect on inner layers of rnn
        rnn_dropout_rate = 0.0 if num_layers == 1 else rnn_dropout_rate
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=rnn_dim,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=False,
            num_layers=num_layers,
            dropout=rnn_dropout_rate,
        )

        # if we're using normalizing flows, instantiate those too
        self.iafs = [
            affine_autoregressive(z_dim, hidden_dims=[iaf_dim]) for _ in range(num_iafs)
        ]
        self.iafs_modules = nn.ModuleList(self.iafs)

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

        self.use_cuda = use_cuda
        # if on gpu cuda-ize all PyTorch (sub)modules
        if use_cuda:
            self.cuda()

    # the model p(x_{1:T} | z_{1:T}) p(z_{1:T})
    def model(
            self,
            mini_batch,
            mini_batch_reversed,
            mini_batch_mask,
            mini_batch_seq_lengths,
            annealing_factor=1.0,
    ):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("dmm", self)

        # set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
        z_prev = self.z_0.expand(mini_batch.size(0), self.z_0.size(0))

        # we enclose all the sample statements in the model in a plate.
        # this marks that each datapoint is conditionally independent of the others
        with pyro.plate("z_minibatch", len(mini_batch)):
            # sample the latents z and observed x's one time step at a time
            # we wrap this loop in pyro.markov so that TraceEnum_ELBO can use multiple samples from the guide at each z
            for t in pyro.markov(range(1, T_max + 1)):
                # the next chunk of code samples z_t ~ p(z_t | z_{t-1})
                # note that (both here and elsewhere) we use poutine.scale to take care
                # of KL annealing. we use the mask() method to deal with raggedness
                # in the observed data (i.e. different sequences in the mini-batch
                # have different lengths)

                # first compute the parameters of the diagonal gaussian distribution p(z_t | z_{t-1})
                z_loc, z_scale = self.trans(z_prev)

                # then sample z_t according to dist.Normal(z_loc, z_scale)
                # note that we use the reshape method so that the univariate Normal distribution
                # is treated as a multivariate Normal distribution with a diagonal covariance.
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(
                        "z_%d" % t,
                        dist.Normal(z_loc, z_scale)
                        .mask(mini_batch_mask[:, t - 1: t])
                        .to_event(1),
                    )

                # compute the probabilities that parameterize the bernoulli likelihood
                emission_loc, emission_scale = self.emitter(z_t)
                # the next statement instructs pyro to observe x_t according to the
                # bernoulli distribution p(x_t|z_t)
                pyro.sample(
                    "obs_x_%d" % t,
                    dist.Normal(emission_loc, emission_scale)
                    .mask(mini_batch_mask[:, t - 1: t])
                    .to_event(1),
                    obs=mini_batch[:, t - 1, :],
                )
                # the latent sampled at this time step will be conditioned upon
                # in the next time step so keep track of it
                z_prev = z_t

    # the guide q(z_{1:T} | x_{1:T}) (i.e. the variational distribution)
    def guide(
            self,
            mini_batch,
            mini_batch_reversed,
            mini_batch_mask,
            mini_batch_seq_lengths,
            annealing_factor=1.0,
    ):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)
        # register all PyTorch (sub)modules with pyro
        pyro.module("dmm", self)

        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        h_0_contig = self.h_0.expand(
            1, mini_batch.size(0), self.rnn.hidden_size
        ).contiguous()
        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)
        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(mini_batch.size(0), self.z_q_0.size(0))

        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("z_minibatch", len(mini_batch)):
            # sample the latents z one time step at a time
            # we wrap this loop in pyro.markov so that TraceEnum_ELBO can use multiple samples from the guide at each z
            for t in pyro.markov(range(1, T_max + 1)):
                # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])

                # if we are using normalizing flows, we apply the sequence of transformations
                # parameterized by self.iafs to the base distribution defined in the previous line
                # to yield a transformed distribution that we use for q(z_t|...)
                if len(self.iafs) > 0:
                    z_dist = TransformedDistribution(
                        dist.Normal(z_loc, z_scale), self.iafs
                    )
                    assert z_dist.event_shape == (self.z_q_0.size(0),)
                    assert z_dist.batch_shape[-1:] == (len(mini_batch),)
                else:
                    z_dist = dist.Normal(z_loc, z_scale)
                    assert z_dist.event_shape == ()
                    assert z_dist.batch_shape[-2:] == (
                        len(mini_batch),
                        self.z_q_0.size(0),
                    )

                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(scale=annealing_factor):
                    if len(self.iafs) > 0:
                        # in output of normalizing flow, all dimensions are correlated (event shape is not empty)
                        z_t = pyro.sample(
                            "z_%d" % t, z_dist.mask(mini_batch_mask[:, t - 1])
                        )
                    else:
                        # when no normalizing flow used, ".to_event(1)" indicates latent dimensions are independent
                        z_t = pyro.sample(
                            "z_%d" % t,
                            z_dist.mask(mini_batch_mask[:, t - 1: t]).to_event(1),
                        )
                # the latent sampled at this time step will be conditioned upon in the next time step
                # so keep track of it
                z_prev = z_t
