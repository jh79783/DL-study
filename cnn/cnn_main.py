from cnn.cnn_model import CnnModel
import numpy as np
import class_model.mode as modelmode
import class_model.mlp_model as mm
import class_model.dataset as dataset
import adam.adam_model as am
import adam.dataset_office31 as officedata

def main():
    fd = dataset.Flower_DataSet([96,96],[96,96,3])
    # data = officedata.Office31Dataset([96,96],[96,96,3])
    mode = modelmode.Select()
    model = CnnModel("flower", fd, mode,[['conv', {'ksize':5, 'chn':6}],
                ['max', {'stride':4}],
                ['conv', {'ksize':3, 'chn':12}],
                ['avg', {'stride':2}]],use_adam=True)
    model.exec_all(epoch_count=5, report=1,batch_size=2)


if __name__ == '__main__':
    # randomize()
    main()