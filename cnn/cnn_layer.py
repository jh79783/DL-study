import cnn.cnn_assist_function as cm
import numpy as np


class Fully:
    def __init__(self, width=None, actfunc='relu'):
        if width is None: pass
        self.output_cnt = width
        self.actfunc = actfunc

    def alloc_layer_param(self, input_shape, rand_std=0.030):
        input_cnt = np.prod(input_shape)
        # print(self.output_cnt)
        # print(rand_std, [input_cnt, self.output_cnt])
        weight = np.random.normal(0, rand_std, [input_cnt, self.output_cnt])
        bias = np.zeros([self.output_cnt])
        # print(weight)
        return {'w': weight, 'b': bias}, [self.output_cnt]

    def forward_layer(self, x, pm):
        # print('fully forward_layer')
        if pm is None: return x, None

        x_org_shape = x.shape

        if len(x.shape) != 2:
            mb_size = x.shape[0]
            x = x.reshape([mb_size, -1])

        affine = np.matmul(x, pm['w']) + pm['b']
        # print(affine)
        y = cm.activate(affine, self.actfunc)
        # print(y)

        return y, [x, y, x_org_shape]

    def backprop_layer(self, G_y, pm, aux):
        if pm is None: return G_y

        x, y, x_org_shape = aux

        G_affine = cm.activate_derv(G_y, y,self.actfunc)

        g_affine_weight = x.transpose()
        g_affine_input = pm['w'].transpose()

        G_weight = np.matmul(g_affine_weight, G_affine)
        G_bias = np.sum(G_affine, axis=0)
        G_input = np.matmul(G_affine, g_affine_input)

        return G_input.reshape(x_org_shape), [G_weight, G_bias]


class Convolution:
    def __init__(self, ksize, chn, actfunc='relu'):
        self.kernels = []
        if ksize is None: pass
        self.ksize = ksize
        self.chn = chn
        self.actfunc = actfunc

    def alloc_layer_param(self, input_shape, rand_std=0.030):
        assert len(input_shape) == 3
        xh, xw, xchn = input_shape
        kh, kw = cm.get_conf_param_2d(self.ksize)
        ychn = self.chn
        # print(kh, kw)
        # print(ychn)
        kernel = np.random.normal(0, rand_std, [kh, kw, xchn, ychn])
        bias = np.zeros([ychn])

        # if self.show_maps: self.kernels.append(kernel)

        return {'k': kernel, 'b': bias}, [xh, xw, ychn]

    def forward_layer(self, x, pm):
        # print('conv forward_layer')
        mb_size, xh, xw, xchn = x.shape
        kh, kw, _, ychn = pm['k'].shape
        # print(kh,kw,ychn)
        x_flat = cm.get_ext_regions_for_conv(x, kh, kw)

        k_flat = pm['k'].reshape([kh * kw * xchn, ychn])
        conv_flat = np.matmul(x_flat, k_flat)
        conv = conv_flat.reshape([mb_size, xh, xw, ychn])
        # y = conv + pm['b']
        y = cm.activate(conv + pm['b'],self.actfunc)
        #
        # if self.need_maps: self.maps.append(y)
        # print(y)

        return y, [x_flat, k_flat, x, y]

    def backprop_layer(self, G_y, pm, aux):
        x_flat, k_flat, x, y = aux

        kh, kw, xchn, ychn = pm['k'].shape
        mb_size, xh, xw, _ = G_y.shape

        G_conv = cm.activate_derv(G_y, y, self.actfunc)

        G_conv_flat = G_conv.reshape(mb_size * xh * xw, ychn)

        g_conv_k_flat = x_flat.transpose()
        g_conv_x_flat = k_flat.transpose()

        G_k_flat = np.matmul(g_conv_k_flat, G_conv_flat)
        G_x_flat = np.matmul(G_conv_flat, g_conv_x_flat)
        G_bias = np.sum(G_conv_flat, axis=0)

        G_kernel = G_k_flat.reshape([kh, kw, xchn, ychn])
        G_input = cm.undo_ext_regions_for_conv(G_x_flat, x, kh, kw)

        return G_input, [G_kernel, G_bias]


class Max_Pooling:
    def __init__(self, stride,actfunc='relu'):
        if stride is None: pass
        self.stride = stride
        self.actfunc = actfunc

    def alloc_layer_param(self, input_shape):
        assert len(input_shape) == 3
        xh, xw, xchn = input_shape
        sh, sw = cm.get_conf_param_2d(self.stride)
        assert xh % sh == 0
        assert xw % sw == 0

        return {}, [xh // sh, xw // sw, xchn]

    def forward_layer(self, x,  pm):
        # print('max forward_layer ')
        mb_size, xh, xw, chn = x.shape
        sh, sw = cm.get_conf_param_2d(self.stride)
        yh, yw = xh // sh, xw // sw

        x1 = x.reshape([mb_size, yh, sh, yw, sw, chn])
        x2 = x1.transpose(0, 1, 3, 5, 2, 4)
        x3 = x2.reshape([-1, sh * sw])

        idxs = np.argmax(x3, axis=1)
        y_flat = x3[np.arange(mb_size * yh * yw * chn), idxs]
        y = y_flat.reshape([mb_size, yh, yw, chn])

        # if self.need_maps: self.maps.append(y)

        return y, idxs

    def backprop_layer(self, G_y,  pm, aux):
        idxs = aux

        mb_size, yh, yw, chn = G_y.shape
        sh, sw = cm.get_conf_param_2d(self.stride)
        xh, xw = yh * sh, yw * sw

        gy_flat = G_y.flatten()

        gx1 = np.zeros([mb_size * yh * yw * chn, sh * sw], dtype='float32')
        gx1[np.arange(mb_size * yh * yw * chn), idxs] = gy_flat[:]
        gx2 = gx1.reshape([mb_size, yh, yw, chn, sh, sw])
        gx3 = gx2.transpose([0, 1, 4, 2, 5, 3])

        G_input = gx3.reshape([mb_size, xh, xw, chn])

        return G_input, None


class Avg_Pooling:
    def __init__(self, stride,actfunc='relu'):
        if stride is None: pass
        self.stride = stride
        self.actfunc = actfunc

    def alloc_layer_param(self, input_shape):
        # print("cnn alloc_pool_layer")
        assert len(input_shape) == 3
        xh, xw, xchn = input_shape
        sh, sw = cm.get_conf_param_2d(self.stride)

        assert xh % sh == 0
        assert xw % sw == 0

        return {}, [xh // sh, xw // sw, xchn]

    def forward_layer(self, x,  pm):
        # print("avg forward_layer")
        mb_size, xh, xw, chn = x.shape
        sh, sw = cm.get_conf_param_2d(self.stride)
        yh, yw = xh // sh, xw // sw

        x1 = x.reshape([mb_size, yh, sh, yw, sw, chn])
        x2 = x1.transpose(0, 1, 3, 5, 2, 4)
        x3 = x2.reshape([-1, sh * sw])

        y_flat = np.average(x3, 1)
        y = y_flat.reshape([mb_size, yh, yw, chn])

        # if self.need_maps: self.maps.append(y)

        return y, None

    def backprop_layer(self, G_y,  pm, aux):
        # print("cnn backprop_avg_layer")
        mb_size, yh, yw, chn = G_y.shape
        sh, sw = cm.get_conf_param_2d(self.stride)
        xh, xw = yh * sh, yw * sw

        gy_flat = G_y.flatten() / (sh * sw)

        gx1 = np.zeros([mb_size * yh * yw * chn, sh * sw], dtype='float32')
        for i in range(sh * sw):
            gx1[:, i] = gy_flat
        gx2 = gx1.reshape([mb_size, yh, yw, chn, sh, sw])
        gx3 = gx2.transpose([0, 1, 4, 2, 5, 3])

        G_input = gx3.reshape([mb_size, xh, xw, chn])

        return G_input , None
