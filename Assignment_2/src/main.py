from helper import create_transformed_loaders
from model import train_model
if __name__ == '__main__':
    batch_size = 64
    learning_rate = 10**-3
    momentum = 0.9
    _, _, train_loader, valid_loader,test_loader = create_transformed_loaders(batch_size)
    train_model(0.5,0.5,300,learning_rate,momentum,"4",train_loader,valid_loader,test_loader)