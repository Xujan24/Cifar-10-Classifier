import pickle

# Function to unpack the cifar-10 dataset.
def unpickle(file):
    with open(file, 'rb') as file:
        data = pickle.load(file, encoding='bytes')
    return data[b'data'], data[b'labels']