from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os
from torch.utils.data import Dataset, Dataloader

class BobberDataset(Dataset):

    def __init__(self, root_dir, indices):
        self.root_dir = root_dir
        self.indices = indices
        with open(os.path.join(root_dir, "target.txt"), 'r') as fhandle:
            labels = map(int, fhandle.read().split('\n')[:-1])
        self.labels = [labels[i] for i in indices]


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        bobbers = np.array([mpimg.imread(os.path.join(root_dir, "{}.png".format(i))) for i in idx])
        return self.bobbers[idx]



def set_loaders(seed = 1, training_set_size = 0.6, validation_set_size = 0.2, test_set_size = 0.2, batch_size = 50):
    num = len(os.listdir('dataset'))
    indices = np.arange(num)
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices = indices[0 : int(num * training_set_size)]
    validation_indices = indices[int(num * training_set_size) : int(num*(training_set_size + validation_set_size))]
    test_indices = indices[int(num*(training_set_size + validation_set_size)):]
    training_set = BobberDataset('dataset', train_indices)
    validation_set = BobberDataset('dataset', validation_indices)
    test_set = BobberDataset('dataset', test_indices)
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader, test_loader
