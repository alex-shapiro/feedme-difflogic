from typing import Literal, final, override

import mlx.core as mx
from mlx import nn

from agent.functional import unique_connections


@final
class LogicLayer(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        grad_factor: float = 1.0,
        connections: Literal["random", "unique"] = "random",
    ):
        super().__init__()
        self.weights = mx.random.normal((d_out, 16))
        self.d_in = d_in
        self.d_out = d_out
        self.grad_factor = grad_factor
        self.connections = connections
        self.n_neurons = d_out
        self.n_weights = d_out
        self.indices = self.get_connections(self.connections)

    @override
    def __call__(self, x: mx.array) -> mx.array:
        assert x.shape[-1] == self.d_in
        a = x[..., self.indices[0]]
        b = x[..., self.indices[1]]

        if self.training:
            w = mx.one_hot(self.weights.argmax(-1), 16)
        # if self.indices

    # TODO: return type
    def get_connections(
        self, connections: Literal["random", "unique"]
    ) -> tuple[mx.array, mx.array]:
        assert self.d_out * 2 >= self.d_in
        if connections == "random":
            # take random input indices to random output indices
            c = mx.random.permutation(2 * self.d_out) % self.d_in
            c = mx.random.permutation(self.d_in)[c]
            c = c.reshape(2, self.d_out)
            a = c[0].astype(mx.int64)
            b = c[1].astype(mx.int64)
            return a, b
        elif connections == "unique":
            return unique_connections(self.d_in, self.d_out)


@final
class GroupSum(nn.Module):
    def __init__(self, k: int, tau: float = 1.0):
        """
        :param k: number of intended real-valued outputs
        :param tau: softmax temperature
        """
        super().__init__()
        self.k = k
        self.tau = tau

    def __call__(self, x: mx.array) -> mx.array:
        assert x.shape[-1] % self.k == 0
        d_head = x.shape[:-1]
        d_tail = x.shape[-1] // self.k
        return x.reshape(*d_head, self.k, d_tail).sum(-1) / self.tau
