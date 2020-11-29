import class_model.mathutil as mu
import numpy as np
import time
from cnn.cnn_layer import Fully
from class_model.model_base import ModelBase

class CnnModel(ModelBase):
    def __init__(self, name, dataset, mode, optimizer, hconfigs,show_maps=False):
        self.show_maps = show_maps
        self.need_maps = False
        self.kernels = []
        super(CnnModel, self).__init__(name, dataset, mode, optimizer)
        self.init_parameters(hconfigs)

    def init_parameters(self, hconfigs):
        self.hconfigs = hconfigs
        self.pm_hiddens = []
        prev_shape = self.dataset.input_shape
        for hconfig in hconfigs.values():
            pm_hidden, prev_shape = hconfig.alloc_layer_param(prev_shape)
            self.pm_hiddens.append(pm_hidden)

        output_cnt = int(np.prod(self.dataset.output_shape))
        self.output_layer = Fully(**{'width': output_cnt})
        self.pm_output, _ = self.output_layer.alloc_layer_param(prev_shape, self.rand_std)

    def alloc_layer_param(self, input_shape, hconfig):
        input_cnt = np.prod(input_shape)
        output_cnt = hconfig
        weight, bias = self.alloc_param_pair([input_cnt, output_cnt])

        return {'w': weight, 'b': bias}, output_cnt

    def alloc_param_pair(self, shape):
        weight = np.random.normal(0, self.rand_std, shape)
        bias = np.zeros([shape[-1]])
        return weight, bias

    def train(self, epoch_count=10, batch_size=10, learning_rate=0.001, report=0):
        self.learning_rate = learning_rate

        batch_count = int(self.dataset.train_count / batch_size)
        time1 = time2 = int(time.time())
        if report != 0:
            print('Model {} train started:'.format(self.name))

        for epoch in range(epoch_count):
            costs = []
            accs = []
            indices = self.dataset.dataset_shuffle_train_data()
            for n in range(batch_count):
                trX, trY = self.dataset.dataset_get_train_data(batch_size, n, indices)
                cost, acc = self.train_step(trX, trY)
                costs.append(cost)
                accs.append(acc)

            if report > 0 and (epoch + 1) % report == 0:
                vaX, vaY = self.dataset.dataset_get_validate_data(100)
                accs_report = []
                for n in range(batch_size):
                    acc_report = self.eval_accuracy(vaX[10 * n:10 * (n + 1)], vaY[10 * n:10 * (n + 1)])
                    accs_report.append(acc_report)
                acc = np.mean(accs_report)
                # acc_org = self.eval_accuracy(vaX, vaY)
                time3 = int(time.time())
                tm1, tm2 = time3 - time2, time3 - time1
                self.dataset.dataset_train_prt_result(epoch + 1, costs, accs, acc, tm1, tm2)
                time2 = time3

        tm_total = int(time.time()) - time1
        print('Model {} train ended in {} secs:'.format(self.name, tm_total))

    def test(self):
        teX, teY = self.dataset.dataset_get_test_data()
        time1 = int(time.time())
        accs_report = []
        for n in range(10):
            acc_report = self.eval_accuracy(teX[10 * n:10 * (n + 1)], teY[10 * n:10 * (n + 1)])
            accs_report.append(acc_report)
        acc = np.mean(accs_report)
        # acc_org = self.eval_accuracy(teX, teY)
        time2 = int(time.time())
        self.dataset.dataset_test_prt_result(self.name, acc, time2 - time1)

    def load_visualize(self, num):
        print('Model {} Visualization'.format(self.name))

        self.need_maps = self.show_maps
        self.maps = []

        deX, deY = self.dataset.dataset_get_validate_data(num)
        est = self.get_estimate(deX)
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



    def train_step(self, x, y):
        self.is_training = True
        output, aux_nn = self.forward_neuralnet(x)
        loss, aux_pp = self.forward_postproc(output, y)
        accuracy = self.eval_accuracy(x, y, output)

        G_loss = 1.0
        G_output = self.backprop_postproc(G_loss, aux_pp)
        self.backprop_neuralnet(G_output, aux_nn)

        self.is_training = False

        return loss, accuracy

    def forward_neuralnet(self, x):
        hidden = x
        aux_layers = []

        for n, hconfig in enumerate(self.hconfigs.values()):
            hidden, aux = hconfig.forward_layer(hidden, self.pm_hiddens[n])
            aux_layers.append(aux)

        output, aux_out = self.output_layer.forward_layer(hidden, self.pm_output)
        return output, [aux_out, aux_layers]

    def backprop_neuralnet(self, G_output, aux):
        aux_out, aux_layers = aux
        G_hidden, G_w_and_b = self.output_layer.backprop_layer(G_output, self.pm_output, aux_out)
        self.optimizer.update_param(self.pm_output, 'w', G_w_and_b[0], self.learning_rate)
        self.optimizer.update_param(self.pm_output, 'b', G_w_and_b[1], self.learning_rate)
        if isinstance(self.hconfigs, dict):
            hconfigs = list(self.hconfigs.values())
        for n in reversed(range(len(hconfigs))):
            hconfig, pm, aux = hconfigs[n], self.pm_hiddens[n], aux_layers[n]
            G_hidden, G_w_and_b = hconfig.backprop_layer(G_hidden, pm, aux)
            keys = list(pm.keys())
            # print(self.pm_hiddens[n].keys())
            if not G_w_and_b is None:
                self.optimizer.update_param(pm, keys[0], G_w_and_b[0], self.learning_rate)
                self.optimizer.update_param(pm, keys[1], G_w_and_b[1], self.learning_rate)

    def forward_postproc(self, output, y):
        loss, aux_loss = self.mode.mode_forward_postproc(output, y)
        extra, aux_extra = self.forward_extra_cost(y)
        return loss + extra, [aux_loss, aux_extra]

    def forward_extra_cost(self, y):
        return 0, None

    def backprop_postproc(self, G_loss, aux):
        aux_loss, aux_extra = aux
        self.backprop_extra_cost(G_loss, aux_extra)
        G_output = self.mode.mode_backprop_postproc(G_loss, aux_loss)
        return G_output

    def backprop_extra_cost(self, G_loss, aux_extra):
        pass

    def eval_accuracy(self, x, y, output=None):
        # print(x.shape)
        if output is None:
            output, _ = self.forward_neuralnet(x)
        # print(output)
        accuracy = self.mode.eval_accuracy(x, y, output)
        # print(accuracy)

        return accuracy

    def get_estimate(self, x):
        output, _ = self.forward_neuralnet(x)
        estimate = self.mode.get_estimate(output)
        return estimate
