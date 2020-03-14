import data_handlers
import cnn_model
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


batch_size = 50
model = torch.load("model.pt").to(device)
training_set, validation_set, test_set = data_handlers.set_loaders(batch_size = batch_size)
accuracy = 0
num = 0
with torch.no_grad():
    running_loss = 0
    for i, data in enumerate(test_set, 0):
        inputs, labels = data[0].float().to(device), data[1].long().to(device)
        outputs = model(inputs)
        accuracy += (np.argmax(outputs.cpu().numpy(), axis = 1) == labels.cpu().numpy()).sum()
        num += len(labels)
print("Accuracy: {:.2f}%".format(100 * accuracy / num))
