# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Conveniance wrapper to perform STFT and iSTFT"""

import torch as th
from .istft import ISTFT


def spectro(x, n_fft=512, hop_length=None, pad=0):
    *other, length = x.shape
    x = x.reshape(-1, length)
    is_mps_xpu = x.device.type in ['mps', 'xpu']
    if is_mps_xpu:
        x = x.cpu()
    z = th.stft(x,
                n_fft * (1 + pad),
                hop_length or n_fft // 4,
                window=th.hann_window(n_fft).to(x),
                win_length=n_fft,
                normalized=True,
                center=True,
                return_complex=False,
                pad_mode='reflect') # Now z will return 1 more dimension - last dimension will be 2 now
    _, freqs, frame, dim = z.shape
    assert dim == 2, "STFT should return complex numbers"

    return z.view(*other, freqs, frame, dim)


def ispectro(z, hop_length=None, length=None, pad=0):
    # B, S, -1, Fr, T  (complex)      ----->     # B, S, -1, Fr, T, 2 shape
    *other, freqs, frames, dim = z.shape   # dim is 2
    print(z.shape)
    assert dim == 2, "iSTFT should receive complex numbers"

    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames, dim)
    win_length = n_fft // (1 + pad)
    is_mps_xpu = z.device.type in ['mps', 'xpu']
    if is_mps_xpu:
        z = z.cpu()
    # x = th.istft(z,
    #              n_fft,
    #              hop_length,
    #              window=th.hann_window(win_length).to(z.real),
    #              win_length=win_length,
    #              normalized=True,
    #              length=length,
    #              center=True)
    x = ISTFT(
        n_fft=n_fft,
        hop_length=hop_length,
        window=th.hann_window(win_length),
        normalized=True)(z)[..., :length]

    _, length = x.shape
    return x.view(*other, length)
