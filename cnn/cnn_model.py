import class_model.mathutil as mu
import time
from cnn.cnn_layer import *


class CnnModel(Fully,Convolution,Max_Pooling,Avg_Pooling):
    # def __init__(self, name, dataset, mode, hconfigs, show_maps=False, use_adam=True):
    #     super(CnnModel, self).__init__(name, dataset, mode, hconfigs, show_maps, use_adam)

    def alloc_layer_param(self, input_shape, hconfig):
        # print("cnn alloc_layer_param")
        layer_type = cm.get_layer_type(hconfig)

        m_name = 'alloc_{}_layer'.format(layer_type)
        method = getattr(self, m_name)
        pm, output_shape = method(input_shape, hconfig)
        return pm, output_shape

    def forward_layer(self, x, hconfig, pm):
        layer_type = cm.get_layer_type(hconfig)
        m_name = 'forward_{}_layer'.format(layer_type)
        method = getattr(self, m_name)
        y, aux = method(x, hconfig, pm)

        return y, aux

    def backprop_layer(self, G_y, hconfig, pm, aux):
        # print("cnn backprop_layer")
        layer_type = cm.get_layer_type(hconfig)

        m_name = 'backprop_{}_layer'.format(layer_type)
        method = getattr(self, m_name)
        G_input = method(G_y, hconfig, pm, aux)

        return G_input

    def activate(self, affine, hconfig):
        if hconfig is None: return affine
        func = cm.get_conf_param(hconfig, 'actfunc', 'relu')

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

    def activate_derv(self, G_y, y, hconfig):
        # print("cnn activate_derv")
        if hconfig is None: return G_y

        func = cm.get_conf_param(hconfig, 'actfunc', 'relu')

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

    def load_visualize(self, num):
        print("cnn visualize")
        print('Model {} Visualization'.format(self.name))

        self.need_maps = self.show_maps
        self.maps = []

        deX, deY = self.dataset.dataset_get_validate_data(num)
        est = self.get_estimate(deX)
        print(self.show_maps)
        if self.show_maps:
            for kernel in self.kernels:
                kh, kw, xchn, ychn = kernel.shape
                grids = kernel.reshape([kh, kw, -1]).transpose(2, 0, 1)
                mu.draw_images_horz(grids[0:5, :, :])

            for pmap in self.maps:
                mu.draw_images_horz(pmap[:, :, :, 0])

        self.dataset.visualize(deX, est, deY)

        self.need_maps = False
        self.maps = None

