from dataset import MushroomDataset
from classifier import Classifier, Models
import numpy as np
from sklearn.preprocessing import OneHotEncoder

if __name__=='__main__':
    dataset = MushroomDataset()
    model = Classifier(model=Models.Adaline)
    ohe = OneHotEncoder(sparse=False)

    data = np.hstack( (np.loadtxt('xor_input.txt', dtype=float).T, np.loadtxt('xor_target.txt', dtype=float).T.reshape(1000,1)) )
    X = data[:,:-1]
    y = ohe.fit_transform(np.atleast_2d(data[:,-1]).T)

    model.train(X, y)
    # model.train(dataset.X, dataset.y)
    print(model.get_stats())
    