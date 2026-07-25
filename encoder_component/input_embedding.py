import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    """
    Converts input token IDs into dense embedding vectors.

    Each token in the vocabulary is mapped to a learnable vector of size
    `d_model` using an embedding lookup table. As described in the
    Transformer paper ("Attention Is All You Need"), the resulting
    embeddings are multiplied by √d_model so that their magnitude is
    comparable to the positional encodings before they are added together.
    This helps provide a more stable scale for the input representations
    during training.
    """

    def __init__(self, d_model: int, vocab_size: int):
        """
        Initializes the input embedding layer.

        Args:
            d_model (int):
                Dimension of each embedding vector (model dimension).
                For example, the original Transformer uses d_model = 512.

            vocab_size (int):
                Total number of unique tokens in the tokenizer's vocabulary.
                This determines the number of rows in the embedding matrix.
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Learnable embedding matrix of shape (vocab_size, d_model)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

    def forward(self, x):
        """
        Converts token IDs into their corresponding embedding vectors.

        Args:
            x (Tensor):
                Tensor containing token IDs with shape
                (batch_size, sequence_length).

        Returns:
            Tensor:
                Tensor of shape (batch_size, sequence_length, d_model)
                containing the scaled token embeddings.

        Note:
            The embedding vectors are multiplied by √d_model, following
            the Transformer paper. Since embedding values are initially
            small, this scaling increases their magnitude so they are on
            a similar scale as the positional encodings before they are
            added together, which helps stabilize training.
        """
        return self.embedding(x) * math.sqrt(self.d_model)