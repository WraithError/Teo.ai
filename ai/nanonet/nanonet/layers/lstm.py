"""
nanonet/layers/lstm.py
Long Short-Term Memory layer — pure NumPy, full BPTT.

Architecture
------------
Four gates computed in one fused matrix multiply for efficiency.
Gate order inside the 4*hidden_size dimension: [i, f, g, o]

    i  = σ( x·Wxi  +  h·Whi  +  bi )   input gate
    f  = σ( x·Wxf  +  h·Whf  +  bf )   forget gate
    g  = tanh( x·Wxg  +  h·Whg  +  bg )   cell gate (candidate)
    o  = σ( x·Wxo  +  h·Who  +  bo )   output gate

    c_t = f ⊙ c_{t-1}  +  i ⊙ g
    h_t = o ⊙ tanh(c_t)
"""

from __future__ import annotations
import numpy as np
from nanonet.core.base import Layer


class LSTM(Layer):
    """
    LSTM layer that slots into any nanonet Sequential model.

    Parameters
    ----------
    input_size       : int
        Number of features at each timestep (last dim of input).
    hidden_size      : int
        Dimensionality of the hidden and cell states.
    return_sequences : bool
        True  → return h at every timestep  (batch, seq_len, hidden_size)
        False → return only the last h       (batch, hidden_size)
        Use True when stacking LSTM layers.
        Use False when feeding into a Dense output layer.

    Input
    -----
    x : ndarray of shape (batch_size, seq_len, input_size)

    Output
    ------
    ndarray of shape (batch_size, seq_len, hidden_size)  if return_sequences
    ndarray of shape (batch_size, hidden_size)            otherwise

    Weights (stored in self.params)
    --------------------------------
    W_x : (input_size,  4 * hidden_size)
    W_h : (hidden_size, 4 * hidden_size)
    b   : (1,           4 * hidden_size)

    Example
    -------
    >>> from nanonet import Sequential, Dense, CrossEntropyLoss, Adam
    >>> from nanonet.layers import LSTM
    >>>
    >>> model = Sequential([
    ...     LSTM(input_size=32, hidden_size=64, return_sequences=False),
    ...     Dense(64, 10, activation="softmax"),
    ... ])
    >>> model.compile(loss=CrossEntropyLoss(), optimizer=Adam(lr=1e-3))
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        return_sequences: bool = False,
    ) -> None:
        super().__init__()

        self.input_size       = input_size
        self.hidden_size      = hidden_size
        self.return_sequences = return_sequences

        H = hidden_size

        # ── Weight initialisation ────────────────────────────────────────────
        # Xavier uniform: keeps gate pre-activations in a reasonable range for
        # sigmoid and tanh.  He would push gates into saturation.
        self.params["W_x"] = self._xavier(input_size, 4 * H)
        self.params["W_h"] = self._xavier(H,          4 * H)
        self.params["b"]   = np.zeros((1, 4 * H))

        self.grads["W_x"] = np.zeros_like(self.params["W_x"])
        self.grads["W_h"] = np.zeros_like(self.params["W_h"])
        self.grads["b"]   = np.zeros_like(self.params["b"])

        self._cache: dict = {}

    # ── Forward pass ────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Run the LSTM over the full sequence.

        x : (batch_size, seq_len, input_size)
        """
        if x.ndim != 3:
            raise ValueError(
                f"LSTM expects 3-D input (batch, seq_len, input_size), "
                f"got shape {x.shape}"
            )

        batch_size, seq_len, _ = x.shape
        H  = self.hidden_size
        Wx = self.params["W_x"]
        Wh = self.params["W_h"]
        b  = self.params["b"]

        # Running states
        h = np.zeros((batch_size, H))   # h_0
        c = np.zeros((batch_size, H))   # c_0

        # Allocate storage for every timestep (needed for BPTT)
        hs        = np.zeros((batch_size, seq_len, H))
        cs        = np.zeros((batch_size, seq_len, H))
        i_gates   = np.zeros((batch_size, seq_len, H))
        f_gates   = np.zeros((batch_size, seq_len, H))
        g_gates   = np.zeros((batch_size, seq_len, H))
        o_gates   = np.zeros((batch_size, seq_len, H))
        h_prev_all = np.zeros((batch_size, seq_len, H))  # h_{t-1} at each t

        for t in range(seq_len):
            x_t = x[:, t, :]               # (batch, input_size)
            h_prev_all[:, t, :] = h        # cache before update

            # Fused gate computation — one GEMM instead of four
            z = x_t @ Wx + h @ Wh + b     # (batch, 4H)

            # Split and apply gate activations
            i_t = self._sigmoid(z[:, 0*H : 1*H])   # input gate
            f_t = self._sigmoid(z[:, 1*H : 2*H])   # forget gate
            g_t = np.tanh(     z[:, 2*H : 3*H])    # cell gate
            o_t = self._sigmoid(z[:, 3*H : 4*H])   # output gate

            # State updates
            c = f_t * c + i_t * g_t        # (batch, H)
            h = o_t * np.tanh(c)            # (batch, H)

            # Store
            hs[:, t, :]      = h
            cs[:, t, :]      = c
            i_gates[:, t, :] = i_t
            f_gates[:, t, :] = f_t
            g_gates[:, t, :] = g_t
            o_gates[:, t, :] = o_t

        self._cache = {
            "x":          x,
            "hs":         hs,
            "cs":         cs,
            "i_gates":    i_gates,
            "f_gates":    f_gates,
            "g_gates":    g_gates,
            "o_gates":    o_gates,
            "h_prev_all": h_prev_all,
            "c0":         np.zeros((batch_size, H)),
        }

        return hs if self.return_sequences else hs[:, -1, :]

    # ── Backward pass (BPTT) ────────────────────────────────────────────────

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backpropagation Through Time.

        grad : (batch, seq_len, H)  if return_sequences=True
               (batch, H)           if return_sequences=False

        Returns
        -------
        dx : (batch, seq_len, input_size)  — gradient w.r.t. the input sequence
        """
        x          = self._cache["x"]
        hs         = self._cache["hs"]
        cs         = self._cache["cs"]
        i_gates    = self._cache["i_gates"]
        f_gates    = self._cache["f_gates"]
        g_gates    = self._cache["g_gates"]
        o_gates    = self._cache["o_gates"]
        h_prev_all = self._cache["h_prev_all"]
        c0         = self._cache["c0"]

        Wx = self.params["W_x"]
        Wh = self.params["W_h"]
        H  = self.hidden_size
        batch_size, seq_len, _ = x.shape

        # Gradient accumulators
        dWx  = np.zeros_like(Wx)
        dWh  = np.zeros_like(self.params["W_h"])
        db   = np.zeros_like(self.params["b"])
        dx   = np.zeros_like(x)

        # ── Distribute grad over timesteps ─────────────────────────────────
        # If not return_sequences, the loss only sees the last h, so only
        # that timestep gets an external gradient.  All other steps receive
        # gradient exclusively through recurrence.
        if not self.return_sequences:
            # grad : (batch, H)
            dh_seq = np.zeros((batch_size, seq_len, H))
            dh_seq[:, -1, :] = grad
        else:
            # grad : (batch, seq_len, H)
            dh_seq = grad

        # Carry-over gradients from future timesteps
        dh_next = np.zeros((batch_size, H))   # dL/dh_{t+1} → dL/dh_t via Wh
        dc_next = np.zeros((batch_size, H))   # dL/dc_{t+1} → dL/dc_t via forget gate

        for t in reversed(range(seq_len)):
            # ── Step 1: total gradient on h_t ───────────────────────────────
            # = direct gradient from output at this step
            # + gradient flowing back from step t+1 through h_{t+1}
            dh = dh_seq[:, t, :] + dh_next     # (batch, H)

            c_t   = cs[:, t, :]
            i_t   = i_gates[:, t, :]
            f_t   = f_gates[:, t, :]
            g_t   = g_gates[:, t, :]
            o_t   = o_gates[:, t, :]
            c_prev = cs[:, t-1, :] if t > 0 else c0
            h_prev = h_prev_all[:, t, :]

            tanh_c = np.tanh(c_t)

            # ── Step 2: gradient through h_t = o_t * tanh(c_t) ─────────────
            do = dh * tanh_c                            # (batch, H)

            # ── Step 3: gradient on c_t ─────────────────────────────────────
            # Sources:
            #   a) through h_t: dh * o_t * tanh'(c_t)
            #   b) from future: dc_next (= dL/dc_{t+1} * f_{t+1})
            dc = dh * o_t * (1.0 - tanh_c ** 2) + dc_next   # (batch, H)

            # ── Step 4: gate gradients ───────────────────────────────────────
            # c_t = f_t * c_{t-1} + i_t * g_t
            df = dc * c_prev                            # (batch, H)
            di = dc * g_t                               # (batch, H)
            dg = dc * i_t                               # (batch, H)

            # ── Step 5: pre-activation gradients ────────────────────────────
            # Chain through each gate's activation function
            di_pre = di * self._sigmoid_grad(i_t)      # sigmoid'
            df_pre = df * self._sigmoid_grad(f_t)      # sigmoid'
            dg_pre = dg * (1.0 - g_t ** 2)            # tanh'
            do_pre = do * self._sigmoid_grad(o_t)      # sigmoid'

            # Concatenate in the same gate order as forward: [i, f, g, o]
            d_gates = np.concatenate([di_pre, df_pre, dg_pre, do_pre], axis=1)
            # d_gates : (batch, 4H)

            # ── Step 6: weight gradients (sum over batch) ────────────────────
            dWx += x[:, t, :].T @ d_gates          # (input_size, 4H)
            dWh += h_prev.T      @ d_gates          # (hidden_size, 4H)
            db  += d_gates.sum(axis=0, keepdims=True)

            # ── Step 7: propagate gradients to previous step ─────────────────
            dx[:, t, :] = d_gates @ Wx.T            # gradient to input at t
            dh_next      = d_gates @ Wh.T            # gradient to h_{t-1}
            dc_next      = dc * f_t                  # gradient to c_{t-1}

        # Write into self.grads so the optimizer can pick them up
        self.grads["W_x"] = dWx
        self.grads["W_h"] = dWh
        self.grads["b"]   = db

        return dx   # (batch, seq_len, input_size)

    # ── Numeric utilities ───────────────────────────────────────────────────

    @staticmethod
    def _xavier(fan_in: int, fan_out: int) -> np.ndarray:
        """Xavier uniform initialisation."""
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid — avoids exp overflow."""
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )

    @staticmethod
    def _sigmoid_grad(s: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid given its output s = σ(x)."""
        return s * (1.0 - s)

    # ── Misc ────────────────────────────────────────────────────────────────

    def num_params(self) -> int:
        return (
            self.params["W_x"].size
            + self.params["W_h"].size
            + self.params["b"].size
        )

    def __repr__(self) -> str:
        return (
            f"LSTM(input_size={self.input_size}, "
            f"hidden_size={self.hidden_size}, "
            f"return_sequences={self.return_sequences})"
        )
