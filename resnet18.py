from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import matplotlib as mpl
import matplotlib.pyplot as plt


def train_model(model, criterion, optimizer, scheduler, num_epochs=1):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    train_loss.append(loss.item() / 15)
                    train_acc.append(torch.sum(preds == labels.data) / 15)
                else:
                    test_loss.append(loss.item() / 15)
                    test_acc.append(torch.sum(preds == labels.data) / 15)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':

    # data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # your image data file
    data_dir = '/opt/data/private/code/color-DFL-CNN/color_dataset/'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'test']}
    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=10,
                                                 shuffle=True,
                                                 num_workers=10) for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    # use gpu or not
    use_gpu = torch.cuda.is_available()

    # get model and replace the original fc layer with your fc layer
    # model_ft = models.resnet18(pretrained=True)
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, 8)
    model_ft = torch.load('/opt/data/private/code/DFL-CNN/resnet.pkl')

    ##paint
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    if use_gpu:
        model_ft = model_ft.cuda()

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=10)

    torch.save(model_ft, '/opt/data/private/code/DFL-CNN/resnet.pkl')
'''
    ##paint
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(train_loss, lw = 1.5, label = 'train_loss')

    plt.subplot(2, 2, 2)
    plt.plot(train_acc, lw = 1.5, label = 'train_acc')

    plt.subplot(2, 2, 3)
    plt.plot(test_loss, lw = 1.5,label = 'loss')

    plt.subplot(2, 2, 4)
    plt.plot(test_acc, lw = 1.5, label = 'acc')
    plt.savefig("resnet18_0.01-10.jpg")
    plt.show()
    print(dataset_sizes)
'''

'''
https://blog.csdn.net/u014380165/article/details/78525273

----------
train Loss: 0.1916 Acc: 0.8083
val Loss: 0.0262 Acc: 0.9778
Epoch 24/24
----------
train Loss: 0.2031 Acc: 0.8250
val Loss: 0.0269 Acc: 1.0000
Training complete in 4m 19s
Best val Acc: 1.000000

'''

'''  lr=0.003
Epoch 9/9
----------
train Loss: 0.1358 Acc: 0.6710
val Loss: 0.1135 Acc: 0.6575
Training complete in 9m 43s
Best val Acc: 0.657500
'''
''' lr=0.01 15
Epoch 9/9
----------
train Loss: 0.0415 Acc: 0.8530
val Loss: 0.0802 Acc: 0.7225
Training complete in 10m 6s
Best val Acc: 0.722500
'''

''' 0.01 10
Epoch 38/39
----------
train Loss: 0.0509 Acc: 0.8640
val Loss: 0.1262 Acc: 0.7325
Epoch 39/39
----------
train Loss: 0.0508 Acc: 0.8520
val Loss: 0.1396 Acc: 0.7200
Training complete in 4m 13s
Best val Acc: 0.737500
'''