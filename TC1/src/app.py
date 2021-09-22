from dataset import MushroomDataset
from nn import NeuralNetwork
import numpy as np
from sklearn.preprocessing import OneHotEncoder

if __name__=='__main__':
    dataset = MushroomDataset()
    nn = NeuralNetwork()
    ohe = OneHotEncoder(sparse=False)

    data = np.hstack( (np.loadtxt('xor_input.txt', dtype=float).T, np.loadtxt('x_or_target.txt', dtype=float).T.reshape(1000,1)) )
    X = data[:,:-1]
    y = ohe.fit_transform(np.atleast_2d(data[:,-1]).T)

    # nn.train(X, y)
    nn.train(dataset.X, dataset.y)
    print(nn.get_stats())
    