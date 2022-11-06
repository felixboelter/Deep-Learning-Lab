#Felix Boelter 2020
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class FFModel(nn.Module):
    def __init__(self, pooldrop, fcdrop):
        """
        We define a convolutional neural network with two convolutional layers, two pooling layers, and
        two fully connected layers. Additionally we define the ReLU activation function and a dropout layer for the pooling and fully connected layers.
        
        :param pooldrop: Dropout rate for the pooling layer
        :param fcdrop: Dropout rate for the fully connected layer
        """
        super(FFModel, self).__init__()
        self.conv1 = nn.Conv2d(3,32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64*5*5,512)
        self.fc2 = nn.Linear(512,10)
        self.activation = nn.ReLU(inplace=True)
        self.drop_pool = nn.Dropout(pooldrop)
        self.drop_fc = nn.Dropout(fcdrop)
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.drop_pool(self.pool1(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.drop_pool(self.pool2(x))
        x = x.reshape(-1, 64 * 5 * 5)
        x = self.drop_fc(self.activation(self.fc1(x)))
        x = self.fc2(x)
        
        return x
def validate_model(model, valid_loader, criterion, device):
    """
    This function validates the model on the validation set.
    
    :param model: The model to be validated
    :param valid_loader: The validation set
    :return: The validation loss and accuracy
    """
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss/len(valid_loader), 100 * correct/total
def train_model(convdrop, fcdrop, num_epochs,learning_rate,momentum, experiment_name, train_loader, valid_loader,test_loader):
    """
    The function takes in the dropout values for the convolutional and fully connected layers, the
    number of epochs, the learning rate, the momentum, the experiment name, the train, validation and
    test loaders. It then creates a model, optimizer and loss function. It then trains the model and
    saves the losses and accuracies for each epoch. It then plots the losses and accuracies for each
    epoch
    
    :param convdrop: dropout rate for convolutional layers
    :param fcdrop: dropout rate for the fully connected layers
    :param num_epochs: number of epochs to train for
    :param learning_rate: The learning rate for the optimizer
    :param momentum: 0.9
    :param experiment_name: a string that will be used to name the model and the plots
    :param train_loader: the training data loader
    :param valid_loader: the validation set
    :param test_loader: the test data loader
    """
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFModel(convdrop,fcdrop)
    model = model.to(device)  # put all model params on GPU.
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum)
    train_losses = []
    val_losses = []
    train_accu = []
    val_accu = []
    for epoch in range(1, num_epochs+1):
        running_loss = 0.0
        running_total = 0
        running_correct = 0
        run_step = 0
        train_loss = []
        train_acc = []
        for i, (images, labels) in enumerate(train_loader):
            model.train()
            images = images.to(device)  
            labels = labels.to(device)  
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            train_loss.append(loss.data.cpu().numpy()) #Append loss value to a local array
            optimizer.zero_grad()  # reset gradients.
            loss.backward()  # compute gradients.
            optimizer.step()  # update parameters.
            
            running_loss += loss.item()
            running_total += labels.size(0)
            model.eval()
            with torch.no_grad():
                _, predicted = outputs.max(1)
            running_correct += (predicted == labels).sum().item()
            run_step += 1
            if i % 100 == 0:
                # check accurary on the batch.
                print(f'epoch: {epoch}, steps: {i}, '
                        f'train_loss: {running_loss / run_step :.3f}, '
                        f'running_acc: {100 * running_correct / running_total:.1f} %')
                
                train_acc.append(100 * running_correct / running_total)
                running_loss = 0.0
                running_total = 0
                running_correct = 0
                run_step = 0
        # validate
        val_loss, val_acc = validate_model(model, valid_loader, loss_fn, device)
        print(f'Validation accuracy: {val_acc} %')
        print('Validation Loss: ', val_loss)
        print(f'Validation error rate: {100 - val_acc: .2f} %')
        
        val_losses.append(val_loss)
        train_losses.append(np.mean(train_loss))
        val_accu.append(val_acc)
        train_accu.append(np.mean(train_acc))
        
    maxval = max(val_accu)
    print('Epoch: ', val_accu.index(maxval)+1, ' Value: ', maxval,'%','loss: ', val_losses[val_accu.index(maxval)])
    print(len(train_losses))
    # Evaluation
    with torch.no_grad():
        correct = 0
        total = 0
        model.eval() 
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device) 
            outputs = model(images)
            _, predicted = outputs.max(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_acc = 100 * correct / total
        print(f'Test accuracy: {test_acc} %')
        print(f'Test error rate: {100 - 100 * correct / total: .2f} %')
    fig, ax1 = plt.subplots()
    fig, ax2 = plt.subplots()
    ax1.plot(train_losses, 'deepskyblue')
    ax1.plot(val_losses, 'coral')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(["Train Loss", "Validation Loss"])
    ax1.title.set_text("Loss Model " + experiment_name)
    ax2.plot(train_accu,'deepskyblue')
    ax2.plot(val_accu,'coral')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend(["Train Accuracy", "Validation Accuracy"])
    ax2.title.set_text("Accuracy Model " + experiment_name)
    plt.show()
    print('Finished Training')
