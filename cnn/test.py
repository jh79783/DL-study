from cnn.cnn_model import CnnModel
import class_model.mode as modelmode
from adam.adam_model import AdamModel
from dataset.dataset_office31 import Office31Dataset
from dataset.dataset import *


def main():
    # fd = Office31Dataset([96, 96], [96, 96, 3])
    # mode = modelmode.Office_Select(fd.cnts)

    fd = Flower_DataSet([96, 96], [96, 96, 3])
    mode = modelmode.Select()
    hconfigs = dict()
    # hconfigs['Fully1'] = Fully(**{'width': 64})
    # hconfigs['Fully2'] = Fully(**{'width': 32})
    # hconfigs['Fully3'] = Fully(**{'width': 10})
    hconfigs['Conv1'] = {'ksize': 3, 'chn': 6}
    hconfigs['Max1'] = {'stride': 2}
    hconfigs['Conv2'] = {'ksize': 3, 'chn': 12}
    hconfigs['Max2'] = {'stride': 2}
    hconfigs['Conv3'] = {'ksize': 3, 'chn': 24}
    hconfigs['Avg1'] = {'stride': 3}

    optimizer = AdamModel(use_adam=True)
    fm3 = CnnModel("f", fd, mode, optimizer, hconfigs)
    fm3.exec_all(epoch_count=10, report=1, learning_rate=0.0001)


if __name__ == '__main__':
    # randomize()
    main()
