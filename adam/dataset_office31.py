from class_model.dataset_base import DatasetBase
import class_model.mathutil as mu
import os
import numpy as np


# noinspection PyCallByClass
class Office31Dataset(DatasetBase):
    def __init__(self, resolution=None, input_shape=None):

        super(Office31Dataset, self).__init__()

        if resolution is None:
            resolution = [100, 100]
        if input_shape is None:
            input_shape = [-1]
        path = '../data/domain_adaptation_images'
        domain_names = mu.list_dir(path)

        images = []
        didxs, oidxs = [], []
        object_names = None

        for dx, dname in enumerate(domain_names):
            domainpath = os.path.join(path, dname, 'images')
            object_names = mu.list_dir(domainpath)

            for ox, oname in enumerate(object_names):
                objectpath = os.path.join(domainpath, oname)
                filenames = mu.list_dir(objectpath)
                for fname in filenames:
                    if fname[-4:] != '.jpg':
                        continue
                    imagepath = os.path.join(objectpath, fname)
                    pixels = mu.load_image_pixels(imagepath, resolution, input_shape)
                    images.append(pixels)
                    didxs.append(dx)
                    oidxs.append(ox)
        self.image_shape = resolution + [3]

        xs = np.asarray(images, np.float32)  # shape(4110, 30000)

        ys0 = mu.onehot(didxs, len(domain_names))  # ys0.shape(4110, 3)
        ys1 = mu.onehot(oidxs, len(object_names))  # ys1.shape(4110, 31)
        ys = np.hstack([ys0, ys1])  # ys.shape(4110, 34)

        self.dataset_shuffle_data(xs, ys, 0.8)
        self.target_names = [domain_names, object_names]
        self.cnts = [len(domain_names)]

    def visualize(self, xs, estimates, answers):
        # print(" office visualize ")
        # print(f"estimates{estimates}\n{answers}")
        mu.draw_images_horz(xs, self.image_shape)
        # print(f"estimates type {type(estimates)} shape {estimates.shape}")
        ests, anss = np.hsplit(estimates, self.cnts), np.hsplit(answers, self.cnts)

        captions = ['도메인', '상품']
        # print(f"self.target_names,{len(self.target_names[0])},\n,{len(self.target_names[1])}")
        for m in range(2):
            print('[ {} 추정결과 ]'.format(captions[m]))
            print(f"ests[{m}]{ests[m].shape}")
            mu.show_select_results(ests[m], anss[m], self.target_names[m], 8)
