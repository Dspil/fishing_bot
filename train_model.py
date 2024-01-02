import data_handlers
import cnn_model
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Working on {}".format(device))
torch.cuda.set_per_process_memory_fraction(0.5, 0)
torch.cuda.empty_cache()

training_losses = []
validation_losses = []

if __name__ == "__main__":
    model = cnn_model.ConvNet().to(device)

    epochs = 5
    batch_size = 8
    learning_rate = 1e-3

    training_set, validation_set, test_set = data_handlers.set_loaders(
        batch_size=batch_size
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = np.inf
    for epoch in range(epochs):
        running_loss = 0
        print(f"Epoch {epoch}:")
        for i, data in tqdm(enumerate(training_set, 0), total=len(training_set)):
            inputs = data[0].float().to(device)
            labels = data[1].float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += np.sqrt(loss.item() / len(inputs))
        training_losses.append(running_loss / len(training_set))
        print("\nEpoch {}, training set loss = {}".format(epoch, running_loss / len(training_set)))

        validation_loss = 0
        print(f"Epoch {epoch} validation:")
        with torch.no_grad():
            for i, data in tqdm(enumerate(validation_set, 0), total=len(validation_set)):
                inputs, labels = data[0].float().to(device), data[1].long().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += np.sqrt(loss.item() / len(inputs))
        validation_losses.append(validation_loss / len(validation_set))
        print("\nEpoch {}, validation set loss = {}".format(epoch, validation_loss / len(validation_set)))
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(model, "model.pt")


print("Report:\n")
print(",".join(f"{i:.4f}" for i in training_losses) + ";" + ",".join(f"{i:.4f}" for i in validation_losses))
model = torch.load("model.pt")
distance = 0
num = 0
with torch.no_grad():
    running_loss = 0
    for i, data in enumerate(validation_set, 0):
        inputs, labels = data[0].float().to(device), data[1].long().to(device)
        outputs = model(inputs)
        distance += ((outputs.cpu().numpy() - labels.cpu().numpy()) ** 2).sum()
        num += len(labels)
print("Accuracy on validation set: {:.2f}%".format(distance / num))
