import class_model.mathutil as mu
import numpy as np
import time
from cnn.cnn_layer import Fully
from class_model.model_base import ModelBase
from cnn.cnn_layer import *


class CnnModel(ModelBase):
    """
    layer를 관리하고 업데이트
    """

    def __init__(self, name, dataset, mode, optimizer, hconfigs, show_maps=False):
        """

        :param name: dataset 이름
        :param dataset: model에 사용될 dataset
        :param mode: dataset에 맞춘 mode
        :param optimizer: optimizer
        :param hconfigs: 은닉층 정보
        :param show_maps: 시각화 여부 결정
        """
        super(CnnModel, self).__init__(name, dataset, mode, optimizer)
        self.show_maps = show_maps
        self.need_maps = False
        self.kernels = []

        self.layers = self.init_layers(hconfigs)

        self.init_parameters()

    def init_layers(self, hconfigs):
        """
        hconfigs을 이용하여 은닉층의 정보 결정 부분
        :param hconfigs: 은닉층 정보의 dictionary
        :return: 은닉층 정보를 가지고 결정한 layer class의 dictionary
        """
        if isinstance(hconfigs, dict):

            layers = dict()
            for name, config in hconfigs.items():
                if name.startswith('Conv'):
                    layers[name] = Convolution(**config)
                if name.startswith('Max'):
                    layers[name] = Max_Pooling(**config)
                if name.startswith('Avg'):
                    layers[name] = Avg_Pooling(**config)
                if name.startswith('Fully'):
                    layers[name] = Fully(**config)
            return layers
        return hconfigs

    def init_parameters(self):
        """
        layer의 파라미터 생성
        """
        # self.pm_hiddens = []
        prev_shape = self.dataset.input_shape
        for layer in self.layers.values():
            prev_shape = layer.alloc_layer_param(prev_shape)

        output_cnt = int(np.prod(self.dataset.output_shape))
        self.output_layer = Fully(**{'width': output_cnt})
        _ = self.output_layer.alloc_layer_param(prev_shape, self.rand_std)

    def alloc_layer_param(self, input_shape, hconfig):
        """
        각 층의 weight와 bias 초기값 결정
        :param input_shape: 입력의 형태
        :param hconfig: 은닉층 정보
        :return: weight와 bias를 dictionary 형태로 반환환하고 출려의 크기도 반환
        """
        input_cnt = np.prod(input_shape)
        output_cnt = hconfig
        weight, bias = self.alloc_param_pair([input_cnt, output_cnt])

        return {'w': weight, 'b': bias}, output_cnt

    def alloc_param_pair(self, shape):
        """
        weight 와 bias 초기 설정
        :param shape: 해당layer weight의 형태
        :return:  해당layer의 weight, bias
        """
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
            accuracies = []
            indices = self.dataset.dataset_shuffle_train_data()
            for n in range(batch_count):
                trainX, trainY = self.dataset.dataset_get_train_data(batch_size, n, indices)
                cost, accuracy = self.train_step(trainX, trainY)
                costs.append(cost)
                accuracies.append(accuracy)

            if report > 0 and (epoch + 1) % report == 0:
                validateX, validateY = self.dataset.dataset_get_validate_data(100)
                acc_reports= []
                for n in range(batch_size):
                    acc_report = self.eval_accuracy(validateX[10 * n:10 * (n + 1)], validateY[10 * n:10 * (n + 1)])
                    acc_reports.append(acc_report)
                acc = np.mean(acc_reports, axis=0)
                # accuracy = self.eval_accuracy(validateX, validateY)
                time3 = int(time.time())
                tm1, tm2 = time3 - time2, time3 - time1
                self.dataset.dataset_train_prt_result(epoch + 1, costs, accuracies, acc, tm1, tm2)
                time2 = time3

        tm_total = int(time.time()) - time1
        print('Model {} train ended in {} secs:'.format(self.name, tm_total))

    def test(self):
        testX, testY = self.dataset.dataset_get_test_data()
        time1 = int(time.time())
        acc_reports = []
        for n in range(10):
            acc_report = self.eval_accuracy(testX[10 * n:10 * (n + 1)], testY[10 * n:10 * (n + 1)])
            acc_reports.append(acc_report)
        acc = np.mean(acc_reports, axis=0)
        # accuracy = self.eval_accuracy(testX, testY)
        time2 = int(time.time())
        self.dataset.dataset_test_prt_result(self.name, acc, time2 - time1)

    def train_step(self, train_x, train_y):
        self.is_training = True
        feature = self.forward_neuralnet(train_x)
        loss, aux_loss = self.mode.mode_forward_postproc(feature, train_y)
        accuracy = self.eval_accuracy(train_x, train_y, feature)

        G_loss = 1.0
        G_output = self.mode.mode_backprop_postproc(G_loss, aux_loss)
        self.backprop_neuralnet(G_output, None)

        self.is_training = False

        return loss, accuracy

    def forward_neuralnet(self, train_x):
        hidden = train_x

        for n, hconfig in enumerate(self.layers.values()):
            hidden = hconfig.forward_layer(hidden)

        feature = self.output_layer.forward_layer(hidden)
        return feature

    def backprop_neuralnet(self, G_output, aux=None):
        G_hidden, out_aux = self.output_layer.backprop_layer(G_output)
        self.optimizer.update_param(self.output_layer.param, 'w', out_aux[0], self.learning_rate)
        self.optimizer.update_param(self.output_layer.param, 'b', out_aux[1], self.learning_rate)
        if isinstance(self.layers, dict):
            layers_list = list(self.layers.values())
            layers_list.reverse()
        for layer in layers_list:
            G_hidden, layer_aux = layer.backprop_layer(G_hidden)
            layer_keys = list(layer.param.keys())
            if layer_aux is not None:
                self.optimizer.update_param(layer.param, layer_keys[0], layer_aux[0], self.learning_rate)
                self.optimizer.update_param(layer.param, layer_keys[1], layer_aux[1], self.learning_rate)

    def eval_accuracy(self, train_x, train_y, feature=None):
        if feature is None:
            feature = self.forward_neuralnet(train_x)
        accuracy = self.mode.eval_accuracy(train_x, train_y, feature)

        return accuracy

    def get_estimate(self, x):
        output = self.forward_neuralnet(x)
        estimate = self.mode.get_estimate(output)
        return estimate

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
