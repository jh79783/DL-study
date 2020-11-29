import numpy as np
import class_model.mode as modelmode
import class_model.mlp_model as mm
import class_model.dataset as dataset
import time
#
# np.random.seed(9999)

# def randomize():
#     np.random.seed(time.time())


def main():

    data = dataset.Pulsar_DataSet()
    mode = modelmode.Binary()
    model = mm.MlpModel("flower", data, mode, [])
    model.exec_all(epoch_count=5, report=1, learning_rate=0.001)


if __name__ == '__main__':
    # randomize()
    main()


