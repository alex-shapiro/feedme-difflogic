from typing import Literal, final, override

import mlx.core as mx
from mlx import nn

import agent.functional as F


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
        self.indices = self.get_connections(connections)

    @override
    def __call__(self, x: mx.array) -> mx.array:
        assert x.shape[-1] == self.d_in
        a = x[..., self.indices[0]]
        b = x[..., self.indices[1]]
        # during training, use use probablistic weights for all binary ops
        # during eval, use the max-probability weight for binary ops
        w = (
            nn.softmax(self.weights)
            if self.training
            else F.one_hot(self.weights.argmax(-1), 16)
        )
        return F.binary_ops(a, b, w)

    def get_connections(
        self, connections: Literal["random", "unique"]
    ) -> tuple[mx.array, mx.array]:
        """
        Maps input => output dimensions.
        Each returned array is an input A or B.
        Each element A[i] or B[i] is an input index that maps to Output[i].
        """
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
            raise NotImplementedError("unique_connections is not yet implemented")
            # return unique_connections(self.d_in, self.d_out)


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
