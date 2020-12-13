import numpy as np
import class_model.mathutil as mu

def get_layer_type(hconfig):
    if not isinstance(hconfig, list): return 'full'
    return hconfig[0]


def get_conf_param(hconfig, key, defval=None):
    if not isinstance(hconfig, list): return defval
    if len(hconfig) <= 1: return defval
    if not key in hconfig[1]: return defval
    return hconfig[1][key]


def get_conf_param_2d(hconfig):
    if isinstance(hconfig, list): return hconfig
    if isinstance(hconfig, int): return [hconfig, hconfig]


def get_ext_regions_for_conv(x, kh, kw):
    mb_size, xh, xw, xchn = x.shape
    regs = get_ext_regions(x, kh, kw)
    regs = regs.transpose([2, 0, 1, 3, 4, 5])

    return regs.reshape([mb_size * xh * xw, kh * kw * xchn])


def get_ext_regions(x, kh, kw):
    mb_size, xh, xw, xchn = x.shape

    eh, ew = xh + kh - 1, xw + kw - 1
    bh, bw = (kh - 1) // 2, (kw - 1) // 2

    x_ext = np.zeros((mb_size, eh, ew, xchn), dtype='float32')
    x_ext[:, bh:bh + xh, bw:bw + xw, :] = x
    regs = np.zeros((xh, xw, mb_size * kh * kw * xchn), dtype='float32')

    for r in range(xh):
        for c in range(xw):
            regs[r, c, :] = x_ext[:, r:r + kh, c:c + kw, :].flatten()

    return regs.reshape([xh, xw, mb_size, kh, kw, xchn])


def undo_ext_regions_for_conv(regs, x, kh, kw):
    mb_size, xh, xw, xchn = x.shape

    regs = regs.reshape([mb_size, xh, xw, kh, kw, xchn])
    regs = regs.transpose([1, 2, 0, 3, 4, 5])

    return undo_ext_regions(regs, kh, kw)


def undo_ext_regions(regs, kh, kw):
    xh, xw, mb_size, kh, kw, xchn = regs.shape

    eh, ew = xh + kh - 1, xw + kw - 1
    bh, bw = (kh - 1) // 2, (kw - 1) // 2

    gx_ext = np.zeros([mb_size, eh, ew, xchn], dtype='float32')

    for r in range(xh):
        for c in range(xw):
            gx_ext[:, r:r + kh, c:c + kw, :] += regs[r, c]

    return gx_ext[:, bh:bh + xh, bw:bw + xw, :]

def activate(affine,hconfig):

    func = get_conf_param(hconfig, 'actfunc', 'relu')
    if func == 'none':
        return affine
    elif func == 'relu':
        return mu.relu(affine)
    elif func == 'sigmoid':
        return mu.sigmoid(affine)
    elif func == 'tanh':
        return mu.tanh(affine)
    else:
        assert 0


def activate_derv(G_y, y, hconfig):
    if hconfig is None: return G_y
    func = get_conf_param(hconfig, 'actfunc', 'relu')

    if func == 'none':
        return G_y
    elif func == 'relu':
        return mu.relu_derv(y) * G_y
    elif func == 'sigmoid':
        return mu.sigmoid_derv(y) * G_y
    elif func == 'tanh':
        return mu.tanh_derv(y) * G_y
    else:
        assert 0