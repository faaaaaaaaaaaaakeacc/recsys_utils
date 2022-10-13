import tf_geometric as tfg
import tensorflow as tf
import numpy as np
import keras


class NGCFHead(keras.Model):
    """Head of NGCF model."""

    def __init__(self,
                 num_ids: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float,
                 ):
        """Init NGCFHead.

        Parameters
        ----------
        num_ids: int
            number of input users/items
        embedding_dim: int
            dimension of embeddings
        hidden_dim: int:
            dimension of hidden state
        
        """
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(num_ids, embedding_dim)
        self.modules = []
        self.dropout = tf.keras.layers.Dropout(dropout)
        for i in range(num_layers):
            self.modules.append(tfg.layers.GCN(hidden_dim))

    def __call__(self, inputs, training=None):
        """ Computes
        """
        x, edge_index, edge_weight = inputs
        hidden = self.embedding(x)
        for layer in self.modules:
            hidden = layer([hidden, edge_index, edge_weight])
        return hidden


class NGCFModule(keras.Model):
    def __init__(self):
        super().__init__()
        pass
