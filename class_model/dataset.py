from class_model.dataset_base import DatasetBase
import class_model.mathutil as mu
import numpy as np
import os


class Abalone_DataSet(DatasetBase):
    def __init__(self):
        super(Abalone_DataSet, self).__init__()
        rows, _ = mu.load_csv('../data/abalone.csv')

        xs = np.zeros([len(rows), 10])
        ys = np.zeros([len(rows), 1])

        for n, row in enumerate(rows):
            if row[0] == 'I':
                xs[n, 0] = 1
            if row[0] == 'M':
                xs[n, 1] = 1
            if row[0] == 'F':
                xs[n, 2] = 1
            xs[n, 3:] = row[1:-1]
            ys[n, :] = row[-1:]

        self.dataset_shuffle_data(xs, ys, 0.8)

    def visualize(self, xs, estimates, answers):
        for n in range(len(xs)):
            x, est, ans = xs[n], estimates[n], answers[n]
            xstr = mu.vector_to_str(x, '%4.2f')
            print('{} => 추정 {:4.1f} : 정답 {:4.1f}'.format(xstr, est[0], ans[0]))


class Pulsar_DataSet(DatasetBase):
    def __init__(self):
        super(Pulsar_DataSet, self).__init__()
        rows, _ = mu.load_csv('../data/pulsar_stars.csv')
        data = np.asarray(rows, dtype='float32')
        self.dataset_shuffle_data(data[:, :-1], data[:, -1:], 0.8)
        self.target_names = ['별', '펄서']

    def visualize(self, xs, estimates, answers):
        for n in range(len(xs)):
            x, est, ans = xs[n], estimates[n], answers[n]
            xstr = mu.vector_to_str(x, '%5.1f', 3)
            estr = self.target_names[int(round(est[0]))]
            astr = self.target_names[int(round(ans[0]))]
            rstr = 'O'
            if estr != astr: rstr = 'X'
            print('{} => 추정 {}(확률 {:4.2f}) : 정답 {} => {}'.format(xstr, estr, est[0], astr, rstr))


class Steel_DataSet(DatasetBase):
    def __init__(self):
        super(Steel_DataSet, self).__init__()
        rows, headers = mu.load_csv('../data/faults.csv')
        data = np.asarray(rows, dtype='float32')
        self.dataset_shuffle_data(data[:, :-7], data[:, -7:], 0.8)

        self.target_names = headers[-7:]

    def visualize(self, xs, estimates, answers):
        mu.show_select_results(estimates, answers, self.target_names)


class Pulsar_Select_DataSet(DatasetBase):
    def __init__(self):
        super(Pulsar_Select_DataSet, self).__init__()
        rows, _ = mu.load_csv('../data/pulsar_stars.csv')
        data = np.asarray(rows, dtype='float32')
        self.dataset_shuffle_data(data[:, :-1], mu.onehot(data[:, -1], 2), 0.8)
        self.target_names = ['별', '펄서']

    def visualize(self, xs, estimates, answers):
        mu.show_select_results(estimates, answers, self.target_names)


class Flower_DataSet(DatasetBase):
    def __init__(self, resolution=None, input_shape=None):
        super(Flower_DataSet, self).__init__()

        if resolution is None:
            resolution = [100, 100]
        if input_shape is None:
            input_shape = [-1]
        path = '../data/flowers'
        self.target_names = mu.list_dir(path)

        images = []
        idxs = []

        for dx, dname in enumerate(self.target_names):
            subpath = path + '/' + dname
            filenames = mu.list_dir(subpath)
            for fname in filenames:
                if fname[-4:] != '.jpg':
                    continue
                imagepath = os.path.join(subpath, fname)
                pixels = mu.load_image_pixels(imagepath, resolution, input_shape)
                images.append(pixels)
                idxs.append(dx)
        self.image_shape = resolution + [3]
        xs = np.asarray(images, np.float32)
        ys = mu.onehot(idxs, len(self.target_names))
        self.dataset_shuffle_data(xs, ys, 0.8)

    def visualize(self, xs, estimates, answers):
        mu.draw_images_horz(xs, self.image_shape)
        mu.show_select_results(estimates, answers, self.target_names)
