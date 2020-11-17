from class_model.model_base import ModelBase
import class_model.mathutil as mu
import numpy as np
import time


class MlpModel(ModelBase):
    def __init__(self, name, dataset, mode, hconfigs):
        super(MlpModel, self).__init__(name, dataset, mode)
        self.init_parameters(hconfigs)

    def init_parameters(self, hconfigs):
        self.hconfigs = hconfigs
        self.pm_hiddens = []
        prev_shape = self.dataset.input_shape
        # print(prev_shape)
        for hconfig in hconfigs:
            pm_hidden, prev_shape = self.alloc_layer_param(prev_shape, hconfig)
            self.pm_hiddens.append(pm_hidden)
        output_cnt = int(np.prod(self.dataset.output_shape))
        print(output_cnt)
        self.pm_output, _ = self.alloc_layer_param(prev_shape, output_cnt)

    def alloc_layer_param(self, input_shape, hconfig):
        input_cnt = np.prod(input_shape)
        output_cnt = hconfig
        weight, bias = self.alloc_param_pair([input_cnt, output_cnt])

        return {'w':weight, 'b':bias}, output_cnt

    def alloc_param_pair(self, shape):
        weight = np.random.normal(0, self.rand_std, shape)
        bias = np.zeros([shape[-1]])
        return weight, bias

    def train(self, epoch_count=10, batch_size=10, learning_rate=0.001, report=0):
        self.learning_rate = learning_rate

        batch_count = int(self.dataset.train_count / batch_size)
        print(batch_size * batch_count)
        time1 = time2 = int(time.time())
        if report != 0:
            print('Model {} train started:'.format(self.name))

        for epoch in range(epoch_count):
            costs = []
            accs = []
            indices = self.dataset.dataset_shuffle_train_data()
            for n in range(batch_count):
                trX, trY = self.dataset.dataset_get_train_data(batch_size, n,indices)
                cost, acc = self.train_step(trX, trY)
                costs.append(cost)
                accs.append(acc)

            if report > 0 and (epoch + 1) % report == 0:
                vaX, vaY = self.dataset.dataset_get_validate_data(100)

                acc = self.eval_accuracy(vaX, vaY)
                time3 = int(time.time())
                tm1, tm2 = time3 - time2, time3 - time1
                self.dataset.dataset_train_prt_result(epoch + 1, costs, accs, acc, tm1, tm2)
                time2 = time3

        tm_total = int(time.time()) - time1
        print('Model {} train ended in {} secs:'.format(self.name, tm_total))

    def test(self):
        teX, teY = self.dataset.dataset_get_test_data()
        time1 = int(time.time())
        acc = self.eval_accuracy(teX, teY)
        time2 = int(time.time())
        self.dataset.dataset_test_prt_result(self.name, acc, time2 - time1)

    def load_visualize(self, num):
        print('Model {} Visualization'.format(self.name))
        deX, deY = self.dataset.dataset_get_validate_data(num)
        est = self.get_estimate(deX)
        self.dataset.visualize(deX, est, deY)


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

        for n, hconfig in enumerate(self.hconfigs):
            hidden, aux = self.forward_layer(hidden, hconfig, self.pm_hiddens[n])
            aux_layers.append(aux)
        output, aux_out = self.forward_layer(hidden, None, self.pm_output)
        return output, [aux_out, aux_layers]

    def backprop_neuralnet(self,G_output, aux):
        aux_out, aux_layers = aux

        G_hidden = self.backprop_layer(G_output, None, self.pm_output, aux_out)

        for n in reversed(range(len(self.hconfigs))):
            hconfig, pm, aux = self.hconfigs[n], self.pm_hiddens[n], aux_layers[n]
            G_hidden = self.backprop_layer(G_hidden, hconfig, pm, aux)

        return G_hidden

    def forward_layer(self, x, hconfig, pm):
        y = np.matmul(x, pm['w']) + pm['b']
        if hconfig is not None:
            y = mu.relu(y)
        return y, [x, y]

    def backprop_layer(self, G_y, hconfig, pm, aux):
        x, y = aux

        if hconfig is not None:
            G_y = mu.relu_derv(y) * G_y

        g_y_weight = x.transpose()
        g_y_input = pm['w'].transpose()

        G_weight = np.matmul(g_y_weight, G_y)
        G_bias = np.sum(G_y, axis=0)
        G_input = np.matmul(G_y, g_y_input)

        pm['w'] -= self.learning_rate * G_weight
        pm['b'] -= self.learning_rate * G_bias

        return G_input

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
        if output is None:
            output, _ = self.forward_neuralnet(x)
        accuracy = self.mode.eval_accuracy(x, y, output)
        return accuracy

    def get_estimate(self, x):
        output, _ = self.forward_neuralnet(x)
        estimate = self.mode.get_estimate(output)
        return estimate