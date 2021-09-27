import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Dataset:
    def __init__(self, name, path):
        self.name    = name
        self.path    = path
        self.raw     = None
        self.X       = None
        self.y       = None
        self.samples = -1
        self.ohe     = OneHotEncoder(sparse=False)

    def standardize(self):
        ### Standardization: (d - mean ) / std        [For each column]
        self.X = (self.X - np.mean(self.X, axis=0)) / np.std(self.X, axis=0)

    def normalization(self):
        ### Normalization  : (d - min) / (max - min)  [For each column]
        self.X = ( self.X - np.min(self.X, axis=0) ) / ( np.max(self.X, axis=0) - np.min(self.X, axis=0) )

    def save_arff(self):
        with open('{}.arff'.format(self.name), 'w') as f:
            f.write('@RELATION {}\n'.format(self.name))
            f.write('\n')
            for i in range(self.X.shape[1]):
                f.write('@ATTRIBUTE {} REAL\n'.format(i+1))
            f.write('@ATTRIBUTE class {' + str(list(map(int, set(self.y.flatten()))))[1:-1] + '}\n')
            f.write('\n')
            f.write('@DATA\n')
            for i in range(self.X.shape[0]):
                f.write(','.join(self.X[i].astype(str))+',{}'.format(np.argmax(self.y[i])))
                f.write('\n')
    
    def save_txt(self):
        with open('{}.txt'.format(self.name), 'w') as f:
            for row_X, row_y in zip(self.X, self.y):
                for r in row_X:
                    f.write(str(r)+',')
                f.write('{}\n'.format(np.argmax(row_y)))
                
    def save_txt_split(self):
        with open('{}_input.txt'.format(self.name), 'w') as f:
            for col in self.X.T:
                for c in col:
                    f.write(str(c)+' ')
                f.write('\n')
        with open('{}_target.txt'.format(self.name), 'w') as f:
            for col in self.y.T:
                for c in col:
                    f.write(str(int(c))+' ')
                f.write('\n')

class MushroomDataset(Dataset):
    def __init__(self, path='mushroom_dataset/agaricus-lepiota.data'):
        super().__init__('mushroom', path)
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
        self.X = self.raw[:,1:]
        self.y = self.ohe.fit_transform(np.atleast_2d(self.raw[:,0]).T) # Convert numerical labels into binary (1-out-of-K) labels

class XORDataset(Dataset):
    def __init__(self, path_input='xor_or_dataset/xor_input.txt', path_target='xor_or_dataset/xor_target.txt'):
        super().__init__('xor', '')
        self.path_input = path_input
        self.path_target = path_target
        self.load_dataset()
        self.standardize()

    def load_dataset(self):
        # Loading dataset
        data = np.hstack( (np.loadtxt(self.path_input, dtype=float).T, np.loadtxt(self.path_target, dtype=float).T.reshape(1000,1)) )
        self.raw = data
        self.X = data[:,:-1]
        self.y = self.ohe.fit_transform(np.atleast_2d(data[:,-1]).T)

class ORDataset(Dataset):
    def __init__(self, path_input='xor_or_dataset/or_input.txt', path_target='xor_or_dataset/or_target.txt'):
        super().__init__('or', '')
        self.path_input = path_input
        self.path_target = path_target
        self.load_dataset()
        self.standardize()

    def load_dataset(self):
        # Loading dataset
        data = np.hstack( (np.loadtxt(self.path_input, dtype=float).T, np.loadtxt(self.path_target, dtype=float).T.reshape(1000,1)) )
        self.raw = data
        self.X = data[:,:-1]
        self.y = self.ohe.fit_transform(np.atleast_2d(data[:,-1]).T)
