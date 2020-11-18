from cnn.cnn_bais_model import CnnBasicModel
import numpy as np
import class_model.mode as modelmode
import class_model.mlp_model as mm
import class_model.dataset as ds
import adam.adam_model as am
import adam.dataset_office31 as do

def main():
    fd = ds.Flower_DataSet([96,96],[96,96,3])
    # data = do.Office31Dataset([96,96],[96,96,3])
    mode = modelmode.Select()
    model = CnnBasicModel("flower", fd, mode,[['full',{'width':30}],['full',{'width':10}]],use_adam=False)
    model.exec_all(epoch_count=5, report=1)


if __name__ == '__main__':
    # randomize()
    main()