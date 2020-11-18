
class Alloc(CnnModelBasic):
    def alloc_full_layer(self, input_shape, hconfig):  #
        print("cnn alloc_full_layer")
        input_cnt = np.prod(input_shape)
        output_cnt = cm.get_conf_param(hconfig, 'width', hconfig)

        weight = np.random.normal(0, self.rand_std, [input_cnt, output_cnt])
        bias = np.zeros([output_cnt])

        return {'w': weight, 'b': bias}, [output_cnt]

    def alloc_conv_layer(self, input_shape, hconfig):
        assert len(input_shape) == 3
        xh, xw, xchn = input_shape
        kh, kw = cm.get_conf_param_2d(hconfig, 'ksize')
        ychn = cm.get_conf_param(hconfig, 'chn')

        kernel = np.random.normal(0, self.rand_std, [kh, kw, xchn, ychn])
        bias = np.zeros([ychn])

        if self.show_maps: self.kernels.append(kernel)

        return {'k': kernel, 'b': bias}, [xh, xw, ychn]

    def alloc_max_layer(self, input_shape, hconfig):
        print("cnn alloc_pool_layer")
        assert len(input_shape) == 3
        xh, xw, xchn = input_shape
        sh, sw = cm.get_conf_param_2d(hconfig, 'stride')

        assert xh % sh == 0
        assert xw % sw == 0

        return {}, [xh // sh, xw // sw, xchn]

    def alloc_avg_layer(self, input_shape, hconfig):
        print("cnn alloc_pool_layer")
        assert len(input_shape) == 3
        xh, xw, xchn = input_shape
        sh, sw = cm.get_conf_param_2d(hconfig, 'stride')

        assert xh % sh == 0
        assert xw % sw == 0

        return {}, [xh // sh, xw // sw, xchn]


class Forward_layer(CnnModelBasic):
    def forward_full_layer(self, x, hconfig, pm):
        print("cnn forward_full_layer")
        if pm is None: return x, None

        x_org_shape = x.shape

        if len(x.shape) != 2:
            mb_size = x.shape[0]
            x = x.reshape([mb_size, -1])

        affine = np.matmul(x, pm['w']) + pm['b']
        y = self.activate(affine, hconfig)

        return y, [x, y, x_org_shape]

    def forward_conv_layer_adhoc(self, x, hconfig, pm):
        print("cnn forward_conv_layer_adhoc")
        mb_size, xh, xw, xchn = x.shape
        kh, kw, _, ychn = pm['k'].shape

        conv = np.zeros((mb_size, xh, xw, ychn))

        for n in range(mb_size):
            for r in range(xh):
                for c in range(xw):
                    for ym in range(ychn):
                        for i in range(kh):
                            for j in range(kw):
                                rx = r + i - (kh - 1) // 2
                                cx = c + j - (kw - 1) // 2
                                if rx < 0 or rx >= xh: continue
                                if cx < 0 or cx >= xw: continue
                                for xm in range(xchn):
                                    kval = pm['k'][i][j][xm][ym]
                                    ival = x[n][rx][cx][xm]
                                    conv[n][r][c][ym] += kval * ival

        y = self.activate(conv + pm['b'], hconfig)

        return y, [x, y]

    def forward_conv_layer_better(self, x, hconfig, pm):
        print("cnn forward_conv_layer_better")
        mb_size, xh, xw, xchn = x.shape
        kh, kw, _, ychn = pm['k'].shape

        conv = np.zeros((mb_size, xh, xw, ychn))

        bh, bw = (kh - 1) // 2, (kw - 1) // 2
        eh, ew = xh + kh - 1, xw + kw - 1

        x_ext = np.zeros((mb_size, eh, ew, xchn))
        x_ext[:, bh:bh + xh, bw:bw + xw, :] = x

        k_flat = pm['k'].transpose([3, 0, 1, 2]).reshape([ychn, -1])

        for n in range(mb_size):
            for r in range(xh):
                for c in range(xw):
                    for ym in range(ychn):
                        xe_flat = x_ext[n, r:r + kh, c:c + kw, :].flatten()
                        conv[n, r, c, ym] = (xe_flat * k_flat[ym]).sum()

        y = self.activate(conv + pm['b'], hconfig)

        return y, [x, y]

    def forward_conv_layer(self, x, hconfig, pm):
        print("cnn forward_conv_layer")
        mb_size, xh, xw, xchn = x.shape
        kh, kw, _, ychn = pm['k'].shape

        x_flat = cm.get_ext_regions_for_conv(x, kh, kw)
        k_flat = pm['k'].reshape([kh * kw * xchn, ychn])
        conv_flat = np.matmul(x_flat, k_flat)
        conv = conv_flat.reshape([mb_size, xh, xw, ychn])

        y = self.activate(conv + pm['b'], hconfig)

        if self.need_maps: self.maps.append(y)

        return y, [x_flat, k_flat, x, y]

    def forward_avg_layer(self, x, hconfig, pm):
        print("cnn forward_avg_layer")
        mb_size, xh, xw, chn = x.shape
        sh, sw = cm.get_conf_param_2d(hconfig, 'stride')
        yh, yw = xh // sh, xw // sw

        x1 = x.reshape([mb_size, yh, sh, yw, sw, chn])
        x2 = x1.transpose(0, 1, 3, 5, 2, 4)
        x3 = x2.reshape([-1, sh * sw])

        y_flat = np.average(x3, 1)
        y = y_flat.reshape([mb_size, yh, yw, chn])

        if self.need_maps: self.maps.append(y)

        return y, None

    def forward_max_layer(self, x, hconfig, pm):
        print("cnn forward_max_layer")
        mb_size, xh, xw, chn = x.shape
        sh, sw = cm.get_conf_param_2d(hconfig, 'stride')
        yh, yw = xh // sh, xw // sw

        x1 = x.reshape([mb_size, yh, sh, yw, sw, chn])
        x2 = x1.transpose(0, 1, 3, 5, 2, 4)
        x3 = x2.reshape([-1, sh * sw])

        idxs = np.argmax(x3, axis=1)
        y_flat = x3[np.arange(mb_size * yh * yw * chn), idxs]
        y = y_flat.reshape([mb_size, yh, yw, chn])

        if self.need_maps: self.maps.append(y)

        return y, idxs


class Backproc_layer(CnnModelBasic):
    def backprop_full_layer(self, G_y, hconfig, pm, aux):
        print("cnn backprop_full_layer")
        if pm is None: return G_y

        x, y, x_org_shape = aux

        G_affine = self.activate_derv(G_y, y, hconfig)

        g_affine_weight = x.transpose()
        g_affine_input = pm['w'].transpose()

        G_weight = np.matmul(g_affine_weight, G_affine)
        G_bias = np.sum(G_affine, axis=0)
        G_input = np.matmul(G_affine, g_affine_input)

        self.update_param(pm, 'w', G_weight)
        self.update_param(pm, 'b', G_bias)

        return G_input.reshape(x_org_shape)

    def backprop_conv_layer(self, G_y, hconfig, pm, aux):
        print("cnn backprop_conv_layer")
        x_flat, k_flat, x, y = aux

        kh, kw, xchn, ychn = pm['k'].shape
        mb_size, xh, xw, _ = G_y.shape

        G_conv = self.activate_derv(G_y, y, hconfig)

        G_conv_flat = G_conv.reshape(mb_size * xh * xw, ychn)

        g_conv_k_flat = x_flat.transpose()
        g_conv_x_flat = k_flat.transpose()

        G_k_flat = np.matmul(g_conv_k_flat, G_conv_flat)
        G_x_flat = np.matmul(G_conv_flat, g_conv_x_flat)
        G_bias = np.sum(G_conv_flat, axis=0)

        G_kernel = G_k_flat.reshape([kh, kw, xchn, ychn])
        G_input = cm.undo_ext_regions_for_conv(G_x_flat, x, kh, kw)

        self.update_param(pm, 'k', G_kernel)
        self.update_param(pm, 'b', G_bias)

        return G_input

    def backprop_max_layer(self, G_y, hconfig, pm, aux):
        print("cnn backprop_max_layer")
        idxs = aux

        mb_size, yh, yw, chn = G_y.shape
        sh, sw = cm.get_conf_param_2d(hconfig, 'stride')
        xh, xw = yh * sh, yw * sw

        gy_flat = G_y.flatten()

        gx1 = np.zeros([mb_size * yh * yw * chn, sh * sw], dtype='float32')
        gx1[np.arange(mb_size * yh * yw * chn), idxs] = gy_flat[:]
        gx2 = gx1.reshape([mb_size, yh, yw, chn, sh, sw])
        gx3 = gx2.transpose([0, 1, 4, 2, 5, 3])

        G_input = gx3.reshape([mb_size, xh, xw, chn])

        return G_input

    def backprop_avg_layer(self, G_y, hconfig, pm, aux):
        # print("cnn backprop_avg_layer")
        mb_size, yh, yw, chn = G_y.shape
        sh, sw = cm.get_conf_param_2d(hconfig, 'stride')
        xh, xw = yh * sh, yw * sw

        gy_flat = G_y.flatten() / (sh * sw)

        gx1 = np.zeros([mb_size * yh * yw * chn, sh * sw], dtype='float32')
        for i in range(sh * sw):
            gx1[:, i] = gy_flat
        gx2 = gx1.reshape([mb_size, yh, yw, chn, sh, sw])
        gx3 = gx2.transpose([0, 1, 4, 2, 5, 3])

        G_input = gx3.reshape([mb_size, xh, xw, chn])

        return G_input