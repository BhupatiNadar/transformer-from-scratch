import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding used in the
    Transformer architecture.

    The Transformer processes all tokens in parallel. Because of this,
    the self-attention mechanism does not automatically know the order
    or position of the tokens.

    Positional encoding adds position information to each token embedding.

    The formulas are:

        PE(pos, 2i) = sin(pos / 10000^(2i / d_model))

        PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))

    Where:

        pos:
            Position of a token in the sequence.

            pos = 0, 1, 2, ..., seq_len - 1

        i:
            Index of a sine-cosine frequency pair.

            i = 0, 1, 2, ...

        2i:
            Even embedding dimensions.

            0, 2, 4, 6, ...

        2i + 1:
            Odd embedding dimensions.

            1, 3, 5, 7, ...

        d_model:
            Total number of dimensions in every token embedding.
    """

    def __init__(
        self,
        d_model: int,
        seq_len: int,
        dropout: float
    ) -> None:
        """
        Creates the complete positional encoding matrix.

        Args:
            d_model (int):
                Size of every token embedding.

                If d_model = 512, every token is represented by
                512 numbers.

                Positional encoding must also contain 512 values for
                each position so that it can be added to the token
                embedding.

            seq_len (int):
                Maximum number of token positions supported by the model.

                Positional encodings will be created for:

                    position 0
                    position 1
                    position 2
                    ...
                    position seq_len - 1

            dropout (float):
                Probability of randomly setting values to zero after
                adding positional encoding to token embeddings.

                Dropout helps reduce overfitting.
        """

        # Call the constructor of nn.Module.
        #
        # This initializes the internal PyTorch functionality required
        # for parameters, buffers, training mode, evaluation mode,
        # device movement and model saving.
        super().__init__()

        # Store the embedding dimension.
        #
        # d_model is the number of values used to represent one token.
        self.d_model = d_model

        # Store the maximum sequence length.
        #
        # Positional encodings will be created from position 0 to
        # position seq_len - 1.
        self.seq_len = seq_len

        # Create a dropout layer.
        #
        # It will be applied after adding positional encoding:
        #
        #     output = Dropout(token_embedding + positional_encoding)
        self.dropout = nn.Dropout(dropout)

        # -------------------------------------------------------------
        # STEP 1: Create the positional encoding matrix
        # -------------------------------------------------------------

        # Create a matrix filled with zeros.
        #
        # Shape:
        #
        #     (seq_len, d_model)
        #
        # Rows represent token positions.
        # Columns represent embedding dimensions.
        #
        # Mathematically:
        #
        #                    embedding dimensions
        #                0      1      2       ...    d_model-1
        #
        # position 0   [ 0      0      0       ...       0 ]
        # position 1   [ 0      0      0       ...       0 ]
        # position 2   [ 0      0      0       ...       0 ]
        #    ...
        #
        # Later, even columns will be filled with sine values and
        # odd columns will be filled with cosine values.
        pe = torch.zeros(seq_len, d_model)

        # -------------------------------------------------------------
        # STEP 2: Create all token-position values
        # -------------------------------------------------------------

        # torch.arange(0, seq_len) creates:
        #
        #     [0, 1, 2, 3, ..., seq_len - 1]
        #
        # These values represent "pos" in the formula:
        #
        #     PE(pos, 2i)
        #
        # Before unsqueeze:
        #
        #     shape = (seq_len,)
        #
        # After unsqueeze(1):
        #
        #     shape = (seq_len, 1)
        #
        # It becomes a column vector:
        #
        #     position =
        #
        #         [[0],
        #          [1],
        #          [2],
        #          ...
        #          [seq_len - 1]]
        #
        # The column shape is necessary for broadcasting.
        # Every position must be multiplied by every frequency value.
        position = torch.arange(
            0,
            seq_len,
            dtype=torch.float
        ).unsqueeze(1)

        # -------------------------------------------------------------
        # STEP 3: Calculate the frequency term
        # -------------------------------------------------------------

        # The paper's formula contains:
        #
        #        1
        # ----------------
        # 10000^(2i/d_model)
        #
        # This code calculates exactly that same value.
        #
        # First:
        #
        #     torch.arange(0, d_model, 2)
        #
        # produces the even dimension indices:
        #
        #     [0, 2, 4, 6, ...]
        #
        # These values represent 2i from the formula.
        #
        # For example:
        #
        #     i = 0  -> 2i = 0
        #     i = 1  -> 2i = 2
        #     i = 2  -> 2i = 4
        #
        # Now consider the complete calculation:
        #
        #     exp(
        #         2i * (-log(10000) / d_model)
        #     )
        #
        # Rearranging:
        #
        #     exp(
        #         -(2i / d_model) * log(10000)
        #     )
        #
        # Using:
        #
        #     exp(a * log(b)) = b^a
        #
        # we obtain:
        #
        #     10000^(-2i / d_model)
        #
        # Using:
        #
        #     a^(-x) = 1 / a^x
        #
        # we obtain:
        #
        #                       1
        #     div_term = ----------------
        #                10000^(2i/d_model)
        #
        # Therefore, this is mathematically identical to the
        # denominator part of the paper's formula.
        #
        # Shape:
        #
        #     (d_model / 2,)
        #
        # There is one frequency value for every sine-cosine pair.
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        # -------------------------------------------------------------
        # STEP 4: Fill even dimensions using sine
        # -------------------------------------------------------------

        # pe[:, 0::2] means:
        #
        #     :     -> select all positions/rows
        #     0::2  -> select columns 0, 2, 4, 6, ...
        #
        # These are the even embedding dimensions represented by 2i.
        #
        # position has shape:
        #
        #     (seq_len, 1)
        #
        # div_term has shape:
        #
        #     (d_model / 2,)
        #
        # Through broadcasting:
        #
        #     position * div_term
        #
        # produces a matrix of shape:
        #
        #     (seq_len, d_model / 2)
        #
        # Each element is:
        #
        #     pos * 1 / 10000^(2i/d_model)
        #
        # which is:
        #
        #     pos / 10000^(2i/d_model)
        #
        # Applying sine gives:
        #
        #     PE(pos, 2i)
        #         =
        #     sin(pos / 10000^(2i/d_model))
        #
        # Therefore, this line directly implements the first formula
        # from the Transformer paper.
        pe[:, 0::2] = torch.sin(position * div_term)

        # -------------------------------------------------------------
        # STEP 5: Fill odd dimensions using cosine
        # -------------------------------------------------------------

        # pe[:, 1::2] means:
        #
        #     :     -> select all positions/rows
        #     1::2  -> select columns 1, 3, 5, 7, ...
        #
        # These are the odd embedding dimensions represented by 2i + 1.
        #
        # The same position * div_term matrix is used because every
        # sine-cosine pair uses the same frequency.
        #
        # For example:
        #
        #     dimensions 0 and 1 use the same frequency
        #     dimensions 2 and 3 use the same frequency
        #     dimensions 4 and 5 use the same frequency
        #
        # Applying cosine gives:
        #
        #     PE(pos, 2i + 1)
        #         =
        #     cos(pos / 10000^(2i/d_model))
        #
        # Therefore, this line directly implements the second formula
        # from the Transformer paper.
        pe[:, 1::2] = torch.cos(position * div_term)

        # -------------------------------------------------------------
        # STEP 6: Add the batch dimension
        # -------------------------------------------------------------

        # Before unsqueeze:
        #
        #     pe.shape = (seq_len, d_model)
        #
        # After unsqueeze(0):
        #
        #     pe.shape = (1, seq_len, d_model)
        #
        # The first dimension is added for the batch.
        #
        # Input embeddings normally have shape:
        #
        #     (batch_size, sequence_length, d_model)
        #
        # Positional encoding has shape:
        #
        #     (1, sequence_length, d_model)
        #
        # During addition, PyTorch broadcasts the positional encoding
        # across every sequence in the batch.
        pe = pe.unsqueeze(0)

        # -------------------------------------------------------------
        # STEP 7: Store pe as a non-trainable model buffer
        # -------------------------------------------------------------

        # Positional encodings are fixed mathematical values.
        # They are not learned during training.
        #
        # Therefore, pe should not be an nn.Parameter.
        #
        # register_buffer stores pe as part of the model so that:
        #
        # 1. It is saved inside the model's state_dict.
        #
        # 2. It automatically moves to the GPU when calling:
        #
        #        model.to("cuda")
        #
        # 3. It automatically moves back to the CPU when required.
        #
        # 4. The optimizer does not update it.
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Adds positional encoding to the input token embeddings.

        Args:
            x (torch.Tensor):
                Input token embeddings.

                Expected shape:

                    (batch_size, current_sequence_length, d_model)

        Returns:
            torch.Tensor:
                Token embeddings containing both semantic information
                and positional information.

                Output shape:

                    (batch_size, current_sequence_length, d_model)
        """

        # x.shape[1] is the length of the current input sequence.
        #
        # The positional encoding matrix may have been created for a
        # larger maximum sequence length.
        #
        # For example:
        #
        #     self.pe shape:
        #         (1, seq_len, d_model)
        #
        #     x shape:
        #         (batch_size, current_sequence_length, d_model)
        #
        # Therefore:
        #
        #     self.pe[:, :x.shape[1], :]
        #
        # selects only the positional encodings required for the current
        # input sequence.
        #
        # Slice meaning:
        #
        #     first ":"        -> keep the batch dimension
        #
        #     :x.shape[1]      -> select positions from 0 up to the
        #                         current sequence length
        #
        #     final ":"        -> select all embedding dimensions
        #
        # Selected positional encoding shape:
        #
        #     (1, current_sequence_length, d_model)
        #
        # Input shape:
        #
        #     (batch_size, current_sequence_length, d_model)
        #
        # PyTorch broadcasts the first dimension:
        #
        #     (batch_size, current_sequence_length, d_model)
        #                        +
        #     (1, current_sequence_length, d_model)
        #
        # Result:
        #
        #     (batch_size, current_sequence_length, d_model)
        #
        # Mathematically, for batch b, position pos and dimension j:
        #
        #     output[b, pos, j]
        #         =
        #     x[b, pos, j] + PE[pos, j]
        #
        # requires_grad_(False) indicates that positional encoding is
        # fixed and should not receive gradient updates.
        #
        # Since pe is already registered as a buffer, it does not require
        # gradients by default. Therefore, requires_grad_(False) is
        # technically optional here.
        x = x + (
            self.pe[:, :x.shape[1], :]
        ).requires_grad_(False)

        # Apply dropout after adding positional encoding.
        #
        # Mathematically:
        #
        #     output = Dropout(x + PE)
        #
        # During training, some values are randomly set to zero.
        # During evaluation, dropout is disabled.
        #
        # The tensor shape does not change.
        return self.dropout(x)