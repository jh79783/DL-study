import numpy as np
import class_model.mode as modelmode
import class_model.mlp_model as mm
import class_model.dataset as dataset
import adam.adam_model as am
import adam.dataset_office31 as do


# np.random.seed(1234)

# def randomize():
#     np.random.seed(time.time())

def main():
    data = do.Office31Dataset()
    mode = modelmode.Office_Select(data.cnts)
    model = am.AdamModel("office31", data, mode, [])
    model.exec_all(epoch_count=5, report=1, learning_rate=0.001)


if __name__ == '__main__':
    # randomize()
    main()
