import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
def create_transformed_loaders(batch_size):
    """
    This function creates the train, validation and test loaders for the CIFAR10 dataset.
    We transform with a normalization using the mean and standard deviation of the CIFAR10 dataset.
    Finally, we create the train, validation loaders by randomly sampling a subset from the train and validation sets.
    
    :param batch_size: The number of images to be passed through the network at once
    :return: the train and test sets, and the train, validation and test loaders.
    """
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    no_transform_train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    idx = np.arange(no_transform_train_set.__len__())

    # Use last 1000 images for validation
    val_indices = idx[50000-1000:]
    train_indices= idx[:-1000]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

    train_loader = torch.utils.data.DataLoader(no_transform_train_set, batch_size=batch_size,
                                                sampler=train_sampler, num_workers=2)

    li = []
    li2 = []
    li3 =[]
    for img, lab in train_loader:
        li.append(img)
    li2 = [np.concatenate(li)[:,i].mean() for i in range(3)]
    li3 = [np.concatenate(li)[:,i].std() for i in range(3)]
    print(f"Images mean: {li2}, Images std: {li3}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((li2[0],li2[1],li2[2]),(li3[0],li3[1],li3[2]))
    ])
    transformed_train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, transform=transform, download=True)

    transformed_test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False,transform=transform)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    test_loader = torch.utils.data.DataLoader(
        dataset=transformed_test_set, batch_size=batch_size, shuffle=False)
    train_loader = torch.utils.data.DataLoader(transformed_train_set, batch_size=batch_size,
                                                sampler=train_sampler, num_workers=2)

    valid_loader = torch.utils.data.DataLoader(transformed_train_set, batch_size=batch_size,
                                                sampler=valid_sampler, num_workers=2)
    return no_transform_train_set, transformed_train_set, train_loader, valid_loader, test_loader