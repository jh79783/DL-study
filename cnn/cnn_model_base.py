from adam.adam_model import AdamModel


class CnnModelBasic(AdamModel):
    def __init__(self, name, dataset, mode, hconfigs, show_maps=False, use_adam=True):
        if isinstance(hconfigs, list) and not isinstance(hconfigs[0], (list, int)):
            hconfigs = [hconfigs]
        self.show_maps = show_maps
        self.need_maps = False
        self.kernels = []
        super(CnnModelBasic, self).__init__(name, dataset, mode, hconfigs, use_adam)

    def alloc_layer_param(self, input_shape, hconfig):
        pass


    def forward_layer(self, x, hconfig, pm):
        pass

    def backprop_layer(self, G_y, hconfig, pm, aux):
        pass

    def activate(self, affine, hconfig):
        pass

    def activate_derv(self, G_y, y, hconfig):
        pass

    def load_visualize(self, num):
        pass