import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import scipy.stats
from enum import Enum

class NNTypes(Enum):
    Adaline  = 'Adaline'
    Logistic = 'Logistic'
    MLP      = 'MLP'

class NeuralNetwork:
    def __init__(self, classifier=NNTypes.Adaline, runs=10, epochs=10, l_rate=0.01, p_train=0.8):
        self.classifier = classifier
        self.W_         = None
        self.runs       = runs
        self.epochs     = epochs
        self.l_rate     = l_rate
        self.p_train    = p_train
        self.n_attrib   = -1
        self.n_labels   = -1
        self.rates      = []
        self.rates_lbs  = []

    def train(self, X, y):
        self.n_attrib = X.shape[1]
        self.n_labels = y.shape[1]
        for loop in range(1, self.runs+1):
            print('Loop: {}'.format(loop))

            # Shuffle rows of the data matrix
            X, y = shuffle(X, y)

            # Split into training and tests subsets
            X_train, y_train = X[:round(self.p_train*len(X))], y[:round(self.p_train*len(y))]
            X_test, y_test   = X[round(self.p_train*len(X)):], y[round(self.p_train*len(y)):]

            # Weights matrix (random init)
            self.W_ = 0.1*np.random.random((self.n_labels, self.n_attrib+1))
            squared_error_epochs = []

            ### Training
            for epoch in range(1, self.epochs):
                print('Epoch: {}'.format(epoch))

                # Shuffle training part
                X_train, y_train = shuffle(X_train, y_train)

                # Fit
                if self.classifier == NNTypes.Adaline:
                    squared_error = self.fit_adaline(X_train, y_train)
                elif self.classifier == NNTypes.Logistic:
                    pass
                elif self.classifier == NNTypes.MLP:
                    pass
                else:
                    print('> Error: No valid classifier')
                    exit(-1)
                squared_error_epochs.append(squared_error/X_train.shape[0]) # Learning Curve

            ### Evaluation
            squared_error = self.evaluation(X_test, y_test)

    def evaluation(self, X, y):
        confusion = np.zeros((self.n_labels,self.n_labels))
        squared_error = 0
        for i in range(X.shape[0]):
            x = np.atleast_2d(np.concatenate(([-1], X[i,:]), axis=0)) # Add bias
            Ui = np.dot(x, self.W_.T) # Predict based on weights matrix
            squared_error += 0.5*np.sum(np.power(y[i,:] - Ui, 2)) # sum of squared errors
            predicted = np.argmax(Ui)
            real = np.argmax(y[i,:])
            confusion[predicted,real]+=1
        rates_lb = []
        for lb in range(self.n_labels):
            TP = confusion[lb,lb]
            FP = np.sum(confusion[lb,:]) - confusion[lb,lb]
            FN = np.sum(confusion[:,lb]) - confusion[lb,lb]
            TN = np.sum(confusion) - TP - FP - FN
            rates_lb.append( (TP+TN)/(TP+TN+FP+FN) )
        self.rates.append(np.sum(np.diag(confusion))/np.sum(confusion))
        self.rates_lbs.append(rates_lb)
        return squared_error/X.shape[0];

    def fit_adaline(self, X, y):
        squared_error = 0
        for i in range(X.shape[0]):
            x = np.atleast_2d(np.concatenate(([-1], X[i,:]), axis=0)) # Add bias
            Ui = np.dot(self.W_, x.T) # Predict based on weights matrix
            error = y[i,:] - Ui.T # Error
            squared_error += 0.5*np.sum(np.power(error, 2)) # sum of squared errors
            self.W_ += self.l_rate * np.dot(error.T, x) # Weights matrix adjustment
        return squared_error

    def predict(self):
        pass

    def get_stats(self):
        stats = {}
        stats['mean']   = np.mean(self.rates)
        stats['std']    = np.std(self.rates)
        stats['median'] = np.median(self.rates)
        stats['min']    = np.min(self.rates)
        stats['max']    = np.max(self.rates)
        stats['mean_lbs'] = np.mean(np.array(self.rates_lbs), axis=0)
        stats['std_lbs'] = np.std(np.array(self.rates_lbs), axis=0)
        return stats