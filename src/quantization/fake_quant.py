"""
src/fake_quant.py

Custom post-training quantization primitives.

Classes
-------
FakeQuantize   -- affine INT8 quantize+dequantize with STE, fixed scale/zero_point
QConv2d        -- drop-in Conv2d replacement with per-tensor weight + activation FQ
QLinear        -- drop-in Linear replacement with per-tensor weight + activation FQ

Design notes
------------
* Scale and zero_point are registered buffers (not nn.Parameters).
  calibrate.py writes them once; nothing ever learns them.
* An `enabled` flag lets sensitivity.py toggle individual layers on/off
  without removing them from the module tree.
* The STE (straight-through estimator) makes round() differentiable:
  forward = round(x), backward = identity.  Needed so that calling
  .backward() on the model for debug purposes doesn't silently produce
  zero gradients.
* Both weight and activation get their own FakeQuantize instance so
  calibrate.py can assign different scales to each.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FakeQuantize
# ---------------------------------------------------------------------------

class FakeQuantize(nn.Module):
    """
    Affine (asymmetric) INT8 fake-quantizer.

    Forward pass:
        q     = clamp(round(x / scale) + zero_point, 0, 255)
        x_hat = (q - zero_point) * scale

    Gradients pass straight through (STE).

    Parameters
    ----------
    n_bits : int
        Bit-width. Default 8 → quint8 range [0, 255].

    Buffers (set by calibrate.py via set_calibration)
    --------------------------------------------------
    scale      : float tensor, shape ()
    zero_point : int tensor,   shape ()

    Attributes
    ----------
    enabled : bool
        When False, forward() is an identity. Lets sensitivity.py
        toggle this layer without touching the module tree.
    calibrated : bool
        Becomes True after set_calibration() is called.
        QConv2d / QLinear will raise if you try to run uncalibrated.
    """

    def __init__(self, n_bits: int = 8) -> None:
        super().__init__()
        self.n_bits = n_bits
        self.q_min  = 0
        self.q_max  = 2 ** n_bits - 1           # 255 for INT8

        # Buffers keep scale/zp on the right device automatically
        self.register_buffer("scale",      torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0))

        self.enabled    = True
        self.calibrated = False

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def set_calibration(self, scale: float, zero_point: int) -> None:
        """Write scale and zero_point after a calibration pass."""
        self.scale.fill_(scale)
        self.zero_point.fill_(zero_point)
        self.calibrated = True

    def calibrate_from_tensor(self, x: torch.Tensor) -> None:
        """
        Convenience: compute and set scale/zp from the min/max of x.
        Useful for weight calibration (called once, offline).
        """
        x_min = float(x.min())
        x_max = float(x.max())
        # Avoid degenerate case where all values are identical
        if x_max == x_min:
            x_max = x_min + 1e-6
        scale     = (x_max - x_min) / (self.q_max - self.q_min)
        zero_point = int(round(-x_min / scale))
        zero_point = int(torch.tensor(zero_point).clamp(self.q_min, self.q_max).item())
        self.set_calibration(scale, zero_point)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        s  = self.scale
        zp = self.zero_point.float()

        # STE: round() on forward, identity on backward
        x_scaled  = x / s
        x_rounded = x_scaled + (torch.round(x_scaled) - x_scaled).detach()

        # Clamp to [q_min, q_max] in integer space, then shift back to float
        q     = torch.clamp(x_rounded + zp, self.q_min, self.q_max)
        x_hat = (q - zp) * s
        return x_hat

    def extra_repr(self) -> str:
        return (
            f"n_bits={self.n_bits}, "
            f"scale={self.scale.item():.6f}, "
            f"zero_point={self.zero_point.item()}, "
            f"enabled={self.enabled}, "
            f"calibrated={self.calibrated}"
        )


# ---------------------------------------------------------------------------
# QConv2d
# ---------------------------------------------------------------------------

class QConv2d(nn.Module):
    """
    Drop-in Conv2d replacement with per-tensor fake-quantization on both
    weights and activations.

    Usage
    -----
    Constructed by quantize.py when replacing Conv2d nodes in a trained model:

        qconv = QConv2d.from_conv2d(conv_layer)
        # then calibrate.py sets qconv.act_fq.set_calibration(s, zp)

    The weight FakeQuantize is calibrated immediately from conv.weight during
    construction (weights are fixed at PTQ time).  The activation FakeQuantize
    must be calibrated externally from a forward pass over calibration data.

    Parameters
    ----------
    All standard Conv2d args are forwarded to an internal nn.Conv2d.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        n_bits: int = 8,
    ) -> None:
        super().__init__()

        # Store the original conv as a sub-module so its parameters are
        # tracked properly by the optimizer / state-dict.
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )

        self.weight_fq = FakeQuantize(n_bits)
        self.act_fq    = FakeQuantize(n_bits)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, n_bits: int = 8) -> "QConv2d":
        """
        Build a QConv2d from an existing Conv2d, copying weights/bias and
        immediately calibrating the weight quantizer from the weight tensor.
        """
        has_bias = conv.bias is not None
        qconv = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=has_bias,
            n_bits=n_bits,
        )
        qconv.conv.weight = conv.weight
        if has_bias:
            qconv.conv.bias = conv.bias

        # Calibrate weights offline from the actual weight tensor
        qconv.weight_fq.calibrate_from_tensor(conv.weight.data)
        return qconv

    # ------------------------------------------------------------------
    # Convenience toggles (used by sensitivity.py)
    # ------------------------------------------------------------------

    def enable_quantization(self) -> None:
        self.weight_fq.enabled = True
        self.act_fq.enabled    = True

    def disable_quantization(self) -> None:
        self.weight_fq.enabled = False
        self.act_fq.enabled    = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_fq.enabled and not self.act_fq.calibrated:
            raise RuntimeError(
                f"{self.__class__.__name__}: activation FakeQuantize has not been "
                "calibrated. Run calibrate.py before inference."
            )

        # Fake-quantize activations then weights, run the conv in FP32 arithmetic
        x_q = self.act_fq(x)
        w_q = self.weight_fq(self.conv.weight)

        return F.conv2d(
            x_q, w_q, self.conv.bias,
            self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups,
        )

    def extra_repr(self) -> str:
        c = self.conv
        return (
            f"{c.in_channels}, {c.out_channels}, "
            f"kernel_size={c.kernel_size}, stride={c.stride}, "
            f"padding={c.padding}"
        )


# ---------------------------------------------------------------------------
# QLinear
# ---------------------------------------------------------------------------

class QLinear(nn.Module):
    """
    Drop-in Linear replacement with per-tensor fake-quantization on both
    weights and activations.  Follows the same design as QConv2d.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        bias: bool = True,
        n_bits: int = 8,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.weight_fq = FakeQuantize(n_bits)
        self.act_fq    = FakeQuantize(n_bits)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_linear(cls, linear: nn.Linear, n_bits: int = 8) -> "QLinear":
        """
        Build a QLinear from an existing Linear, copying weights/bias and
        immediately calibrating the weight quantizer.
        """
        has_bias = linear.bias is not None
        qlinear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            n_bits=n_bits,
        )
        qlinear.linear.weight = linear.weight
        if has_bias:
            qlinear.linear.bias = linear.bias

        qlinear.weight_fq.calibrate_from_tensor(linear.weight.data)
        return qlinear

    # ------------------------------------------------------------------
    # Convenience toggles (used by sensitivity.py)
    # ------------------------------------------------------------------

    def enable_quantization(self) -> None:
        self.weight_fq.enabled = True
        self.act_fq.enabled    = True

    def disable_quantization(self) -> None:
        self.weight_fq.enabled = False
        self.act_fq.enabled    = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_fq.enabled and not self.act_fq.calibrated:
            raise RuntimeError(
                f"{self.__class__.__name__}: activation FakeQuantize has not been "
                "calibrated. Run calibrate.py before inference."
            )

        x_q = self.act_fq(x)
        w_q = self.weight_fq(self.linear.weight)

        return F.linear(x_q, w_q, self.linear.bias)

    def extra_repr(self) -> str:
        ln = self.linear
        return f"in_features={ln.in_features}, out_features={ln.out_features}"