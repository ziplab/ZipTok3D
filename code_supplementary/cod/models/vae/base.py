from abc import ABC, abstractmethod

from torch import nn


class BaseAutoencoder(nn.Module, ABC):
    embed_dim: int = -1

    def encode(self, pc, **kwargs):
        z = self.encode_embed(pc)
        return self.encode_latents(z, **kwargs)

    def decode(self, z, mask=None, **kwargs):
        z, mask = self.decode_latents(z, mask=mask, **kwargs)
        return self.decode_embed(z, mask=mask, **kwargs)

    @abstractmethod
    def encode_embed(self, pc):
        raise NotImplementedError

    @abstractmethod
    def encode_latents(self, z, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def decode_latents(self, z, mask=None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def decode_embed(self, z, mask=None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def decode_queries(self, context, queries):
        raise NotImplementedError

    def load_autoencoder_weights(self, state_dict):
        pass
