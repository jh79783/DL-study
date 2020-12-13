import cnn.cnn_assist_function as cm
import numpy as np


class Fully:
    def __init__(self, width=None, actfunc='relu'):
        if width is None: pass
        self.output_cnt = width
        self.actfunc = actfunc
        self.param = dict()
        self.aux = None

    def alloc_layer_param(self, input_shape, rand_std=0.030):
        input_cnt = np.prod(input_shape)
        weight = np.random.normal(0, rand_std, [input_cnt, self.output_cnt])
        bias = np.zeros([self.output_cnt])
        self.param = {'w': weight, 'b': bias}
        return [self.output_cnt]

    def forward_layer(self, layer_input):
        if self.param is None: return layer_input, None

        layer_input_org_shape = layer_input.shape

        if len(layer_input.shape) != 2:
            mb_size = layer_input.shape[0]
            layer_input = layer_input.reshape([mb_size, -1])
        # Y = X*W + b, (mb_size,output_cnt) = (mb_size,input_cnt)(input_cnt,output_cnt) + (mb_size,output_cnt)
        affine = np.matmul(layer_input, self.param['w']) + self.param['b']
        # affine activate 함수 적용(defval:relu)
        layer_output = cm.activate(affine, self.actfunc)
        self.aux = (layer_input, layer_output, layer_input_org_shape)

        return layer_output

    def backprop_layer(self, G_layer_output):
        """

        :param G_output: 각 층의 dL/dY
        :return:
        """
        if self.param is None: return G_layer_output

        layer_input, layer_output, layer_input_org_shape = self.aux

        # G_Y activate 역함수 적용(defval:relu)
        G_affine = cm.activate_derv(G_layer_output, layer_output, self.actfunc)
        # X_T(input_cnt,mb_size)
        layer_input_trans = layer_input.transpose()
        # W_T(output_cnt,input_cnt)
        w_t = self.param['w'].transpose()
        # G_W = X_T * G_Y, (intput_cnt, output_cnt) = (input_cnt,mb_size)(mb_size,output_cnt)
        G_weight = np.matmul(layer_input_trans, G_affine)
        G_bias = np.sum(G_affine, axis=0)
        # G_X = G_Y * W_T, (mb_size, input_cnt) = (mb_size,output_cnt)(output_cnt,input_cnt)
        G_input = np.matmul(G_affine, w_t)

        return G_input.reshape(layer_input_org_shape), [G_weight, G_bias]


class Convolution:
    def __init__(self, ksize, chn, actfunc='relu'):
        if ksize is None: pass
        self.ksize = ksize
        self.chn = chn
        self.actfunc = actfunc
        self.param = dict()
        self.aux = None

    def alloc_layer_param(self, input_shape, rand_std=0.030):
        assert len(input_shape) == 3
        xh, xw, xchn = input_shape
        kh, kw = cm.get_conf_param_2d(self.ksize)
        ychn = self.chn
        kernel = np.random.normal(0, rand_std, [kh, kw, xchn, ychn])
        bias = np.zeros([ychn])

        # if self.show_maps: self.kernels.append(kernel)
        self.param = {'k': kernel, 'b': bias}
        return [xh, xw, ychn]

    def forward_layer(self, layer_input):
        mb_size, xh, xw, xchn = layer_input.shape
        kh, kw, _, ychn = self.param['k'].shape
        # F_flat (mb_size * xh * xw, kh * kw * xchn)
        feature = cm.get_ext_regions_for_conv(layer_input, kh, kw)
        # K_flat (kh * kw * xchn, ychn)
        kernel_flat = self.param['k'].reshape([kh * kw * xchn, ychn])
        # C_flat = X_flat * K_flat (mb_size * xh * xw, ychn)
        conv_flat = np.matmul(feature, kernel_flat)
        conv = conv_flat.reshape([mb_size, xh, xw, ychn])
        # Y = conv + param['b']
        layer_output = cm.activate(conv + self.param['b'], self.actfunc)

        # if self.need_maps: self.maps.append(y)
        self.aux = (feature, kernel_flat, layer_input, layer_output)
        return layer_output

    def backprop_layer(self, G_layer_output):
        layer_feature_flat, kernel_flat, layer_input, layer_output = self.aux

        kh, kw, xchn, ychn = self.param['k'].shape
        mb_size, xh, xw, _ = G_layer_output.shape

        G_conv = cm.activate_derv(G_layer_output, layer_output, self.actfunc)

        G_conv_flat = G_conv.reshape(mb_size * xh * xw, ychn)
        # dc/dk = K_T(output_cnt,input_cnt)
        g_conv_k_flat = layer_feature_flat.transpose()
        # dc/dx = X_T(input_cnt,mb_size)
        g_conv_x_flat = kernel_flat.transpose()

        # G_K = X_T * G_Y, (intput_cnt, output_cnt) = (input_cnt,mb_size)(mb_size,output_cnt)
        G_k_flat = np.matmul(g_conv_k_flat, G_conv_flat)
        # G_X = G_Y 8 K_T
        G_intput_flat = np.matmul(G_conv_flat, g_conv_x_flat)
        G_bias = np.sum(G_conv_flat, axis=0)

        G_kernel = G_k_flat.reshape([kh, kw, xchn, ychn])
        G_layer_input = cm.undo_ext_regions_for_conv(G_intput_flat, layer_input, kh, kw)

        return G_layer_input, [G_kernel, G_bias]

    # def update_param(self, pm, key, delta,learning_rate = 0.001):
    #     if delta is not None:  # True 이면 아담 업데이트 시작
    #         delta = self.eval_adam_delta(self.param, key, delta)
    #     pm[key] -= learning_rate * delta


class Max_Pooling:
    def __init__(self, stride, actfunc='relu'):
        if stride is None: pass
        self.stride = stride
        self.actfunc = actfunc
        self.param = dict()
        self.aux = None

    def alloc_layer_param(self, input_shape):
        assert len(input_shape) == 3
        xh, xw, xchn = input_shape
        sh, sw = cm.get_conf_param_2d(self.stride)
        assert xh % sh == 0
        assert xw % sw == 0

        return [xh // sh, xw // sw, xchn]

    def forward_layer(self, layer_input):
        mb_size, xh, xw, chn = layer_input.shape
        sh, sw = cm.get_conf_param_2d(self.stride)
        yh, yw = xh // sh, xw // sw

        x1 = layer_input.reshape([mb_size, yh, sh, yw, sw, chn])
        x2 = x1.transpose(0, 1, 3, 5, 2, 4)
        x3 = x2.reshape([-1, sh * sw])

        idxs = np.argmax(x3, axis=1)
        layer_output_flat = x3[np.arange(mb_size * yh * yw * chn), idxs]
        layer_output = layer_output_flat.reshape([mb_size, yh, yw, chn])

        # if self.need_maps: self.maps.append(y)
        self.aux = idxs
        return layer_output

    def backprop_layer(self, G_y):
        idxs = self.aux

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
    def __init__(self, stride, actfunc='relu'):
        if stride is None: pass
        self.stride = stride
        self.actfunc = actfunc
        self.param = dict()
        self.aux = None

    def alloc_layer_param(self, input_shape):
        assert len(input_shape) == 3
        xh, xw, xchn = input_shape
        sh, sw = cm.get_conf_param_2d(self.stride)

        assert xh % sh == 0
        assert xw % sw == 0

        return [xh // sh, xw // sw, xchn]

    def forward_layer(self, x):
        mb_size, xh, xw, chn = x.shape
        sh, sw = cm.get_conf_param_2d(self.stride)
        yh, yw = xh // sh, xw // sw

        x1 = x.reshape([mb_size, yh, sh, yw, sw, chn])
        x2 = x1.transpose(0, 1, 3, 5, 2, 4)
        x3 = x2.reshape([-1, sh * sw])

        y_flat = np.average(x3, 1)
        y = y_flat.reshape([mb_size, yh, yw, chn])

        # if self.need_maps: self.maps.append(y)

        return y

    def backprop_layer(self, G_y):
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

        return G_input, None
