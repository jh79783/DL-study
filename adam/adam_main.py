import class_model.mode as modelmode
import class_model.mlp_model as mm
from adam.adam_model import AdamModel
import dataset.dataset_office31 as do


# np.random.seed(1234)

# def randomize():
#     np.random.seed(time.time())

def main():
    # data =dataset.Flower_DataSet()
    # mode = modelmode.Select()
    data = do.Office31Dataset()
    mode = modelmode.Office_Select(data.cnts)
    optimizer = AdamModel(use_adam=True)
    model = mm.MlpModel("office31", data, mode, [64,32,10],optimizer)
    model.exec_all(epoch_count=10, report=2, learning_rate=0.001)


if __name__ == '__main__':
    # randomize()
    main()
