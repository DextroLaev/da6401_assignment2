from partA.dataset import load_dataset
from partA.model import PretrainedModel

if __name__ == '__main__':
    train_data,val_data,test_data = load_dataset('../inaturalist_12K',batch_size=16)
    model = PretrainedModel(train_method='gradual',train_last_k_layers=3,train_new_fc_step=5)
    model.train_network(train_data,val_data,test_data,10)