import class_model.mode as modelmode
import class_model.mlp_model as mm
from adam.adam_model import AdamModel
import dataset.dataset_office31 as do


# np.random.seed(1234)

# def randomize():
#     np.random.seed(time.time())

def main():
    data = do.Office31Dataset()
    mode = modelmode.Office_Select(data.cnts)
    optimizer = AdamModel()
    model = mm.MlpModel("office31", data, mode, [],optimizer)
    model.exec_all(epoch_count=5, report=1, learning_rate=0.001)


if __name__ == '__main__':
    # randomize()
    main()
