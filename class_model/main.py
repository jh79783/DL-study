import numpy as np
import class_model.mode as modelmode
import class_model.mlp_model as mm
import class_model.dataset as dataset
import time

np.random.seed(9999)

# def randomize():
#     np.random.seed(time.time())


def main():

    data = dataset.Flower_DataSet()
    mode = modelmode.Select()
    model = mm.MlpModel("abalone_model", data, mode, [30,10])
    model.exec_all(epoch_count=10, report=2, learning_rate=0.001)


if __name__ == '__main__':
    # randomize()
    main()


