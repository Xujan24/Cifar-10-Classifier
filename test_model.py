import torch
import numpy as np

from utils import unpickle
from model import CNNModel

def main():
    trained_model = './trained_model.pth'
    test_batch_dir = './cifar-10/test_batch'
    
    classifier = CNNModel()
    classifier.load_state_dict(torch.load(trained_model))
    classifier.cuda()
    classifier.eval()

    test_x, test_y = unpickle(test_batch_dir)
    test_x, test_y = torch.tensor(np.reshape(test_x, (len(test_x), 3, 32, 32))).to('cuda', dtype=torch.float), torch.tensor(test_y).cuda()

    classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    # calculating the accuracy of our classifier;
    print("Calculating accuracy...")
    correct = 0
    total = len(test_x)

    with torch.no_grad():
        out = classifier(test_x)
        _, predicted = torch.max(out, 1)

        # calculate the total accuracy
        correct += (predicted == test_y).sum().item()
        print('Accuracy: %5d %%' % (correct / total * 100))


if __name__ == '__main__':
    main()
