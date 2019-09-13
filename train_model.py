import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from load_data import LoadTrainingData
from model import CNNModel

def train_model(model, data, epoch, batch_size):
    # define the loss function and back propagation algorithm
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.01)

    for e in range(epoch):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print('[EPOCH: %d, Learning Rate: %f]' % (e+1, lr))
        print()
        for i, dataset in enumerate(data):
            inputs, lbl = dataset
            inputs, lbl = inputs.view(batch_size, 3, 32, 32).to('cuda', dtype=torch.float), lbl.view(-1).to('cuda')

            if torch.cuda.is_available():
                inputs, lbl = inputs.cuda(), lbl.cuda()

            # set the gradient for each parameters zero
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, lbl)
            loss.backward()
            optimizer.step()
            print('-[step: %d, loss: %f]' % (i+1, loss.item()))
        scheduler.step()


    print ('Finished Training')


if __name__ == '__main__':
    cnn = CNNModel()
    batch = 2000
    if torch.cuda.is_available():
        cnn.cuda()

    trainingDataset = LoadTrainingData()
    dataLoader = DataLoader(
        dataset=trainingDataset,
        batch_size=batch,
        shuffle=True,
        num_workers=2
    )

    train_model(cnn, dataLoader, epoch=40, batch_size=batch)

    # save model
    torch.save(cnn.state_dict(), './trained_model.pth')
