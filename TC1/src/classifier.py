import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.utils import shuffle
from enum import Enum

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Models(Enum):
    Adaline  = 'Adaline'
    Logistic = 'Logistic'
    LMQ      = 'LMQ'
    MLP      = 'MLP'

class Classifier:
    def __init__(self, model=Models.Adaline, runs=1, epochs=50, n_hidden=10, l_rate=0.1, p_train=0.8, log=False):
        self.model      = model
        self.W_         = None
        self.H_         = None
        self.mom        = 0.0
        self.runs       = runs
        self.epochs     = epochs
        self.n_hidden   = n_hidden
        self.l_rate     = l_rate
        self.p_train    = p_train
        self.log        = log
        self.n_attrib   = -1
        self.n_labels   = -1
        self.l_curve    = [] # learning curve (squared erros by epochs)
        self.rates      = [] # loop rates
        self.rates_lbs  = [] # loop rates by class
        self.lcurv_plot = plt.figure()

        self.X, self.y  = None, None

    def progress_bar(self, run, epoch):
        bar, done = 50, ((run-1)*self.runs+(epoch)+1)/(self.runs*self.epochs)
        sys.stdout.write("\r Training: [%s%s]" % ('=' * int(bar*done), ' ' * (bar-int(bar*done))) )  
        sys.stdout.flush()

    def train(self, X, y):
        self.__init__(self.model, self.runs, self.epochs, self.n_hidden, self.l_rate, self.p_train, self.log)
        self.n_attrib = X.shape[1]
        self.n_labels = y.shape[1]
        self.X, self.y  = X, y
        if self.log:
            print('> Model {}'.format(self.model.name))
        for run in range(1, self.runs+1):
            # Shuffle rows of the data matrix
            X, y = shuffle(X, y)

            # Split into training and tests subsets
            X_train, y_train = X[:round(self.p_train*len(X))], y[:round(self.p_train*len(y))]
            X_test, y_test   = X[round(self.p_train*len(X)):], y[round(self.p_train*len(y)):]

            # Weights matrix (random init)
            if self.model == Models.MLP:
                self.H_ = 0.1*np.random.random((self.n_labels, self.n_hidden+1))
                self.W_ = 0.1*np.random.random((self.n_hidden, self.n_attrib+1))
            else:
                self.W_ = 0.1*np.random.random((self.n_labels, self.n_attrib+1))
            self.l_curve = []

            ### Training
            for epoch in range(1, self.epochs):
                # Progress bar
                if self.log:
                    self.progress_bar(run, epoch)

                # Shuffle training part
                X_train, y_train = shuffle(X_train, y_train)

                # Fit
                if self.model == Models.Adaline:
                    squared_error = self.fit_adaline(X_train, y_train)
                elif self.model == Models.Logistic:
                    squared_error = self.fit_logistic(X_train, y_train)
                elif self.model == Models.LMQ:
                    self.fit_lmq(X_train, y_train)
                    break
                elif self.model == Models.MLP:
                    squared_error = self.fit_mlp(X_train, y_train)
                else:
                    print('> Error: No valid classifier')
                    exit(-1)

                if self.model != Models.LMQ:
                    self.l_curve.append(squared_error/X_train.shape[0]) # Learning Curve

            ### Evaluation
            squared_error = self.evaluation(X_test, y_test)

        # Return
        self.progress_bar(self.runs-1, self.epochs)
        print()
        return self.W_

    def predict(self, sample):
        if self.model != Models.LMQ:
            x = np.atleast_2d(np.concatenate(([-1], sample), axis=0)) # Add bias
            Ui = np.dot(self.W_, x.T) # Predict based on weights matrix (hidden layer)
            Y = self.activation_function(Ui) # Activation function
            if self.model == Models.MLP:
                z = np.concatenate((np.array([[-1]]), Y), axis=0) # Add bias
                Uk = np.dot(self.H_, z) # Predict based on weights matrix (output layer)
                Y = self.activation_function(Uk) # Activation function
            return Y
        else:
            return np.dot(sample, self.W_) # LMQ predict

    def fit_adaline(self, X, y):
        squared_error = 0
        for i in range(X.shape[0]):
            # Prediction
            x = np.atleast_2d(np.concatenate(([-1], X[i,:]), axis=0)) # Add bias
            Ui = np.dot(self.W_, x.T) # Predict based on weights matrix
            Yi = self.activation_function(Ui) # Activation function
            # Error Adjustment
            error = y[i,:] - Yi.T # Error
            squared_error += 0.5*np.sum(np.power(error, 2)) # sum of squared errors
            self.W_ += self.l_rate * np.dot(error.T, x) # Weights matrix adjustment
        return squared_error

    def fit_logistic(self, X, y):
        squared_error = 0
        for i in range(X.shape[0]):
            # Prediction
            x = np.atleast_2d(np.concatenate(([-1], X[i,:]), axis=0)) # Add bias
            Ui = np.dot(self.W_, x.T) # Predict based on weights matrix
            Yi = self.activation_function(Ui) # Activation function
            # Error Adjustment
            error = y[i,:] - Yi.T # Error
            squared_error += 0.5*np.sum(np.power(error, 2)) # sum of squared errors
            derivative = 0.5*(1 - np.power(Yi, 2)) + 0.05 # sigmoid logistic derivative
            gradient = error * derivative.T # local gradient
            self.W_ += self.l_rate * np.dot(gradient.T, x) # Weights matrix adjustment
        return squared_error

    def fit_lmq(self, X, y):
        # Compute the weight matrix (N_attribs, N_samples)x(N_samples, N_labels) = (N_attribs, N_labels)
        self.W_ = np.dot(np.linalg.pinv(X), y)
        return

    def fit_mlp(self, X, y):
        squared_error = 0
        for i in range(X.shape[0]):
            ### Prediction
            # Hidden Layer
            x = np.atleast_2d(np.concatenate(([-1], X[i,:]), axis=0)) # Add bias
            Ui = np.dot(self.W_, x.T) # Predict based on weights matrix (hidden layer)
            Yi = self.activation_function(Ui) # Activation function
            # Output Layer
            z = np.concatenate((np.array([[-1]]), Yi), axis=0) # Add bias
            Uk = np.dot(self.H_, z) # Predict based on weights matrix (output layer)
            Yk = self.activation_function(Uk) # Activation function
            ### Error Adjustment (Backpropagation)
            # Error
            error = y[i,:] - Yk.T # Error
            squared_error += 0.5*np.sum(np.power(error, 2)) # sum of squared errors
            # Local Gradient
            derivative = Yk*(1 - Yk) + 0.01 # sigmoid logistic derivative
            gradient_output = error * derivative.T # local gradient (output)
            derivative = Yi*(1 - Yi) + 0.01 # sigmoid logistic derivative
            gradient_hidden = derivative * np.dot(self.H_[:,1:].T, gradient_output.T) # local gradient (hidden)
            # Weights Adjustment (Output Layer)
            self.H_ += self.l_rate * np.dot(gradient_output.T, z.T) # Weights matrix adjustment
            # Weights Adjustment (Hidden Layer)
            self.W_ += self.l_rate * np.dot(gradient_hidden, x) # Weights matrix adjustment
        return squared_error

    def activation_function(self, Ui):
        if self.model == Models.Adaline:
            return Ui
        elif self.model == Models.Logistic:
            return (1 - np.exp(-Ui))/(1 + np.exp(-Ui)) 
        elif self.model == Models.MLP:
            return 1/(1 + np.exp(-Ui)) # Logistic between [0,1]

    def evaluation(self, X, y):
        confusion = np.zeros((self.n_labels,self.n_labels))
        squared_error = 0
        for i in range(X.shape[0]):
            Y = self.predict(X[i,:])
            squared_error += 0.5*np.sum(np.power(y[i,:] - Y, 2)) # sum of squared errors
            predicted = np.argmax(Y)
            real = np.argmax(y[i,:])
            confusion[predicted,real]+=1
        self.compute_confusion_matrix(confusion)
        return squared_error/X.shape[0];

    def compute_confusion_matrix(self, confusion):
        rates_lb = []
        for lb in range(self.n_labels):
            TP = confusion[lb,lb]
            FP = np.sum(confusion[lb,:]) - confusion[lb,lb]
            FN = np.sum(confusion[:,lb]) - confusion[lb,lb]
            TN = np.sum(confusion) - TP - FP - FN
            rates_lb.append( (TP+TN)/(TP+TN+FP+FN) )
        self.rates.append(np.sum(np.diag(confusion))/np.sum(confusion))
        self.rates_lbs.append(rates_lb)
        # if self.log:
        #     print('confusion matrix: \n{}'.format(confusion))
        #     print('acc: {}'.format(self.rates[-1]))
        return
    
    def get_stats(self):
        stats = {}
        stats['name']     = self.model.name
        stats['runs']     = self.runs
        stats['epochs']   = self.epochs
        stats['n_hidden'] = self.n_hidden
        stats['l_rate']   = self.l_rate
        stats['p_train']  = self.p_train
        stats['mean']     = np.mean(self.rates)
        stats['std']      = np.std(self.rates)
        stats['median']   = np.median(self.rates)
        stats['min']      = np.min(self.rates)
        stats['max']      = np.max(self.rates)
        stats['mean_lbs'] = np.mean(np.array(self.rates_lbs), axis=0)
        stats['std_lbs']  = np.std(np.array(self.rates_lbs), axis=0)
        return stats
    
    def save_stats(self):
        stats = self.get_stats()
        with open('{}.json'.format(stats['name']), 'w') as json_file:
            json.dump(stats, json_file, cls=NumpyEncoder, indent=4, separators=(',', ': '))

    def plot_learning_curve(self):
        if len(self.l_curve)>0:
            plt.figure()
            plt.plot(self.l_curve)
            plt.title('Learning Curve ({})'.format(self.model.name))
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error (MSE)')
            plt.show()
        return
    
    def save_learning_curve(self):
        if len(self.l_curve)>0:
            plt.figure()
            plt.plot(self.l_curve)
            plt.title('Learning Curve ({})'.format(self.model.name))
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error (MSE)')
            plt.savefig('{}.png'.format(self.model.name))
        return

    def save_all_learning_curves(self):
        if len(self.l_curve)>0:
            plt.figure(1)
            l,=plt.plot(self.l_curve)
            plt.title('Learning Curves')
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error (MSE)')
            l.set_label('{}'.format(self.model.name))
            plt.legend()
            plt.savefig('learning_curves.png')
        return

    def plot_2d_decision_surface(self):
        if self.X.shape[1] != 2:
            print('Error: Number of attributes must be exactly 2')
            return
        interval = 0.01
        attr1 = np.arange(np.min(self.X[:,0]), np.max(self.X[:,0]), interval)
        attr2 = np.arange(np.min(self.X[:,1]), np.max(self.X[:,1]), interval)
        lb1, lb2 = [], []
        for a1 in attr1:
            for a2 in attr2:
                if np.argmax(self.predict(np.array([a1, a2])))==0:
                    lb1.append(np.array([a1, a2]))
                else:
                    lb2.append(np.array([a1, a2]))
        lb1, lb2 = np.array(lb1), np.array(lb2)
        # Surface plot
        plt.figure()
        plt.title('Decision Surface ({})'.format(self.model.name))
        if len(lb1)>0:
            plt.scatter(lb1[:,0], lb1[:,1])
        else:
            plt.scatter(self.X[0,0], self.X[0,1])
        if len(lb2)>0:    
            plt.scatter(lb2[:,0], lb2[:,1])
        else:
            plt.scatter(self.X[0,0], self.X[0,1])
        # Samples plot
        for lb in [0,1]:
            cond = np.where(self.y[:,lb]==1)
            plt.scatter(self.X[cond][:,0],self.X[cond][:,1], alpha=0.5)
        plt.show()
