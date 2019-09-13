import os
from torch.utils.data import Dataset
from utils import unpickle


class LoadTrainingData(Dataset):
    def __init__(self):
        self.trainX = []
        self.trainY = []

        data_dir = './cifar-10/training batches'
        batches = os.listdir(data_dir)

        for batch in batches:
            batch_data, batch_labels = unpickle(os.path.join(data_dir, batch))
            self.trainX.extend(batch_data)
            self.trainY.extend(batch_labels)

    def __getitem__(self, item):
        return self.trainX[item], self.trainY[item]

    def __len__(self):
        return len(self.trainX)
