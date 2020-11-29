from cnn.cnn_model import CnnModel
import numpy as np
import class_model.mode as modelmode
import class_model.mlp_model as mm
import class_model.dataset as dataset
from adam.adam_model import AdamModel
import adam.dataset_office31 as officedata
from cnn.cnn_layer import *


def main():
    fd = officedata.Office31Dataset([96, 96], [96, 96, 3])
    mode = modelmode.Office_Select(fd.cnts)
    hconfigs = dict()
    # hconfigs['Fully1'] = Fully(**{'width': 64})
    # hconfigs['Fully2'] = Fully(**{'width': 32})
    # hconfigs['Fully3'] = Fully(**{'width': 10})
    hconfigs['Conv1'] = Convolution(**{'ksize': 3, 'chn': 6})
    hconfigs['Max1'] = Max_Pooling(**{'stride': 2})
    hconfigs['Conv2'] = Convolution(**{'ksize': 3, 'chn': 12})
    hconfigs['Max2'] = Max_Pooling(**{'stride': 2})
    hconfigs['Conv3'] = Convolution(**{'ksize': 3, 'chn': 24})
    hconfigs['Avg1'] = Avg_Pooling(**{'stride': 3})
    optimizer = AdamModel(use_adam=True)
    fm3 = CnnModel("f", fd, mode, optimizer, hconfigs)
    fm3.exec_all(epoch_count=10, report=2, learning_rate=0.001)


if __name__ == '__main__':
    # randomize()
    main()
