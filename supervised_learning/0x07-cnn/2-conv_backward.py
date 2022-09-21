#!/usr/bin/env python3
"""Pooling Forward propagation."""
import numpy as np

def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Backward propagation for Conv layer."""
    m, prev_h, prev_w, _ = A_prev.shape
    m, new_h, new_w, new_c = dZ.shape
    sh, sw = stride
    kh, kw, _, new_c = W.shape
    if padding == 'valid':
        ph, pw = 0, 0
    else:
        ph = int(np.ceil((sh*(prev_h-1)-prev_h+kh)/2))
        pw = int(np.ceil((sw*(prev_w-1)-prev_w+kw)/2))
    npad = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    A_prev = np.pad(A_prev, pad_width=npad, mode='constant')
    dw = np.zeros_like(W)
    dA = np.zeros_like(A_prev)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for img in range(m):
        for h in range(new_h):
            for w in range(new_w):
                x = h * sh
                y = w * sw
                for f in range(new_c):
                    filt = W[:, :, :, f]
                    dz = dZ[img, h, w, f]
                    slice_A = A_prev[img, x:x+kh, y:y+kw,:]
                    dw[:, :, :, f] += slice_A * dz
                    dA[img, x:x+kh, y:y+kw,:] += dz * filt
                    
    if padding == 'same':
        dA = dA[:, ph:-ph, pw:-pw, :]
    return dA, dw, db
