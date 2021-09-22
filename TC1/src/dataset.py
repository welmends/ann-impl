import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

class MushroomDataset:
    def __init__(self, path='mushroom_dataset/agaricus-lepiota.data'):
        self.path    = path
        self.raw     = None
        self.X       = None
        self.y       = None
        self.samples = -1
        self.ohe     = OneHotEncoder(sparse=False)
        self.load_dataset()
        self.standardize()

    def load_dataset(self):
        # Loading dataset (mushroom)
        self.raw = np.loadtxt(self.path, delimiter=',', dtype=str) # load
        attr = [{val: int(cnt) for cnt, val in enumerate(list(set(self.raw[:,i])))} for i in range(len(self.raw[0,:]))] # dict mapping
        self.raw = np.array([np.array([attr[i][e] for e in self.raw[:,i]]) for i in range(len(self.raw[0,:]))]).T # swap str to int

        # Data preparation (Check columns)
        cols = [i for i,e in enumerate(attr) if len(e)==1]
        for col in cols:
            self.raw = np.delete(self.raw, col, axis=1)

        # Data fill
        self.X = self.raw[:,0:]
        self.y = self.ohe.fit_transform(np.atleast_2d(self.raw[:,0]).T) # Convert numerical labels into binary (1-out-of-K) labels

    def standardize(self):
        ### Standardization: (d - mean ) / std        [For each column]
        self.X = (self.X - np.mean(self.X, axis=0)) / np.std(self.X, axis=0)

    def normalization(self):
        ### Normalization  : (d - min) / (max - min)  [For each column]
        self.X = ( self.X - np.min(self.X, axis=0) ) / ( np.max(self.X, axis=0) - np.min(self.X, axis=0) )

    def save_arff(self):
        with open('mushroom.arff', 'w') as f:
            f.write('@RELATION mushroom\n')
            f.write('\n')
            for i in range(self.X.shape[1]):
                f.write('@ATTRIBUTE {} REAL\n'.format(i+1))
            f.write('@ATTRIBUTE class {0,1}\n')
            f.write('\n')
            f.write('@DATA\n')
            for i in range(self.X.shape[0]):
                f.write(','.join(self.X[i].astype(str))+',{}'.format(np.argmax(self.y[i])))
                f.write('\n')
