import numpy as np


class DatasetBase:
    def __init__(self):
        self.tr_xs = np.array([])
        self.tr_ys = np.array([])
        self.va_xs = np.array([])
        self.va_ys = np.array([])
        self.te_xs = np.array([])
        self.te_ys = np.array([])

    def __str__(self):
        return '({}+{}+{})'.format(len(self.tr_xs), len(self.te_xs), len(self.va_xs))

    def dataset_get_test_data(self):
        return self.te_xs, self.te_ys

    def dataset_shuffle_train_data(self,size):
        self.indices = np.arange(size)
        np.random.shuffle(self.indices)

    def dataset_get_train_data(self, batch_size, nth):
        from_idx = nth * batch_size
        to_idx = (nth + 1) * batch_size

        tr_X = self.tr_xs[self.indices[from_idx:to_idx]]
        tr_Y = self.tr_ys[self.indices[from_idx:to_idx]]
        return tr_X, tr_Y

    def dataset_get_validate_data(self, count):
        indices = np.arange(len(self.va_xs))
        np.random.shuffle(indices)
        va_X = self.va_xs[indices[0:count]]
        va_Y = self.va_ys[indices[0:count]]
        return va_X, va_Y

    def dataset_shuffle_data(self, xs, ys, tr_ratio=0.8, va_ratio=0.05):
        data_count = len(xs)

        tr_cnt = int(data_count * tr_ratio / 10) * 10
        va_cnt = int(data_count * va_ratio)
        te_cnt = data_count - (tr_cnt + va_cnt)

        tr_from, tr_to = 0, tr_cnt
        va_from, va_to = tr_cnt, tr_cnt + va_cnt
        te_from, te_to = tr_cnt + va_cnt, data_count

        indices = np.arange(data_count)
        np.random.shuffle(indices)

        self.tr_xs = xs[indices[tr_from:tr_to]]
        self.tr_ys = ys[indices[tr_from:tr_to]]
        self.va_xs = xs[indices[va_from:va_to]]
        self.va_ys = ys[indices[va_from:va_to]]
        self.te_xs = xs[indices[te_from:te_to]]
        self.te_ys = ys[indices[te_from:te_to]]

        self.input_shape = xs[0].shape
        self.output_shape = ys[0].shape

    def visualize(self, xs, estimates, answers):
        pass

    def dataset_train_prt_result(self, epoch, costs, accs, acc, time1, time2):
        print('    Epoch {}: cost={:5.3f}, accuracy={:5.3f}/{:5.3f} ({}/{} secs)'.format(epoch, np.mean(costs),
                                                                                         np.mean(accs), acc, time1,
                                                                                         time2))

    def dataset_test_prt_result(self, name, acc, time):
        print('Model {} test report: accuracy = {:5.3f}, ({} secs)\n'.format(name, acc, time))

    @property
    def train_count(self):
        return len(self.tr_xs)
