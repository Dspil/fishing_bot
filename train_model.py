import data_handlers
import cnn_model
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    
    
    model = cnn_model.ConvNet().to(device)
    training_set, validation_set, test_set = data_handlers.set_loaders(batch_size = batch_size)

    epochs = 60
    batch_size = 50
    learning_rate = 1e-4

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(epochs):
        running_loss = 0
        for i, data in enumerate(training_set, 0):
            inputs, labels = data[0].float().to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print("Epoch {}, loss = {}".format(epoch, running_loss))
            




