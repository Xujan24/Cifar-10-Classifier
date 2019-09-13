## A PyTorch implementation of a simple Convolutional Neural Network (CNN) for Cifar-10 image classification.
In recent years, neural networks have gained a lot of attentions and has been extensively used in various problems, including computer vision,
time-series analysis, Natural language Processing, and so on. A CNN is a class of feedforward neural networks which is used in 
almost all kind of computer vision problems, for example image classification, object detection and tracking, image segmentation,
and so on. In this project, I tried to implement a simple CNN for an image classification task. I have used the Cifar-10 dataset 
for this purpose. More details about the dataset can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## Network Architecture:
The CNN based classifier has two convolutional layers followed by a fully connected layer and an output layer. The first convolutional 
layer has 32 channels and the second one has 64 channels. Each of these convolutional layers are followed by Batch Normalization layer 
and max pooling layer. Rectifier Linear Unit (ReLU) has been used as the activation function and the Batch normalization is done 
after the activation function is applied on them. The final output layer has 10 neurons one for each of the classes.

## Tranining Procedure:
For the cost function, I have employed the cross entropy loss and the network is trained using Stochastic Gradient Descent (SGD) with a 
momentum of 0.9. The network is trained for 40 epoch and for first 20 epoch the learning rate is 
set to 0.2 and for rest, the network is trained with a learning rate of 0.002.

## Results:
I was able to gain an accuracy of 76% on the test dataset. I have included the trained model for your reference. Also the script 
for training and testing the model.
