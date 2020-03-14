import data_handlers
import cnn_model
import torch
import torch.nn as nn
import numpy as np
import sys
from matplotlib import pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Working on {}".format(device))

if __name__ == "__main__":
    
    model = cnn_model.ConvNet().to(device)

    epochs = 10
    batch_size = 100
    learning_rate = 1e-4

    training_set, validation_set, test_set = data_handlers.set_loaders(batch_size = batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    best_loss = np.inf
    for epoch in range(epochs):
        running_loss = 0
        for i, data in enumerate(training_set, 0):
            inputs, labels = data[0].float().to(device), data[1].long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print("\nEpoch {}, training set loss = {}".format(epoch, running_loss))

        validation_loss = 0
        with torch.no_grad():
            for i, data in enumerate(validation_set, 0):
                inputs, labels = data[0].float().to(device), data[1].long().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss
        print("\nEpoch {}, validation set loss = {}".format(epoch, validation_loss))
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(model, "model.pt")
            


model = torch.load("model.pt")
accuracy = 0
num = 0
with torch.no_grad():
    running_loss = 0
    for i, data in enumerate(validation_set, 0):
        inputs, labels = data[0].float().to(device), data[1].long().to(device)
        outputs = model(inputs)
        accuracy += (np.argmax(outputs.cpu().numpy(), axis = 1) == labels.cpu().numpy()).sum()
        num += len(labels)
print("Accuracy on validation set: {:.2f}%".format(100 * accuracy / num))
    



