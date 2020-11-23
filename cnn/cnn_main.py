from cnn.cnn_model import CnnModel
import numpy as np
import class_model.mode as modelmode
import class_model.mlp_model as mm
import class_model.dataset as dataset
import adam.adam_model as am
import adam.dataset_office31 as officedata


def main():
    fd = dataset.Flower_DataSet([96, 96], [96, 96, 3])
    mode = modelmode.Select()

    # fm1 = CnnModel("flower", fd, mode,[30,10],use_adam=True)
    # fm1.exec_all(epoch_count=10, report=2)
    #
    # fm2 = CnnModel("flower", fd, mode,
    #                [['full', {'width': 30}],
    #                 ['full', {'width': 10}]],
    #                use_adam=True)
    # fm2.exec_all(epoch_count=10, report=2, learning_rate=0.001)

    fm3  = CnnModel("flower", fd, mode,[['conv', {'ksize':5, 'chn':6}],
                ['max', {'stride':4}],
                ['conv', {'ksize':3, 'chn':12}],
                ['avg', {'stride':2}]],use_adam=True,show_maps=True)
    fm3.exec_all(epoch_count=10, report=2)

    # fm4 = CnnModel('flowers_model_4', fd,mode,
    #                [['conv', {'ksize':3, 'chn':6}],
    #                 ['max', {'stride':2}],
    #                 ['conv', {'ksize':3, 'chn':12}],
    #                 ['max', {'stride':2}],
    #                 ['conv', {'ksize':3, 'chn':24}],
    #                 ['avg', {'stride':3}]])
    # fm4.exec_all(epoch_count = 10, report = 2)
    #
    # od = officedata.Office31Dataset([96, 96], [96, 96, 3])
    # mode = modelmode.Office_Select(od.cnts)
    #
    # om1 = CnnModel('office31_model_1', od, mode,
    #                [['conv', {'ksize': 3, 'chn': 6}],
    #                 ['max', {'stride': 2}],
    #                 ['conv', {'ksize': 3, 'chn': 12}],
    #                 ['max', {'stride': 2}],
    #                 ['conv', {'ksize': 3, 'chn': 24}],
    #                 ['avg', {'stride': 3}]])
    # om1.exec_all(epoch_count=10, report=2)


if __name__ == '__main__':
    # randomize()
    main()
