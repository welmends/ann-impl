import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import scipy.linalg
import scipy.stats
from enum import Enum

class Models(Enum):
    Adaline  = 'Adaline'
    Logistic = 'Logistic'
    LMQ      = 'LMQ'
    MLP      = 'MLP'

class Classifier:
    def __init__(self, model=Models.Adaline, runs=10, epochs=10, l_rate=0.01, p_train=0.8):
        self.model = model
        self.W_         = None
        self.W_old      = None
        self.mom        = 0.0
        self.runs       = runs
        self.epochs     = epochs
        self.l_rate     = l_rate
        self.p_train    = p_train
        self.n_attrib   = -1
        self.n_labels   = -1
        self.l_curve    = [] # learning curve (squared erros by epochs)
        self.rates      = [] # loop rates
        self.rates_lbs  = [] # loop rates by class

    def train(self, X, y):
        self.__init__(self.model, self.runs, self.epochs, self.l_rate, self.p_train)
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
            self.l_curve = []

            ### Training
            for epoch in range(1, self.epochs):
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
            if self.model != Models.LMQ:
                squared_error = self.evaluation_nn(X_test, y_test)
            else:
                self.evaluation_lmq(X_test, y_test)

        # Return
        return self.W_

    def predict(self):
        pass

    def fit_adaline(self, X, y):
        squared_error = 0
        for i in range(X.shape[0]):
            x = np.atleast_2d(np.concatenate(([-1], X[i,:]), axis=0)) # Add bias
            Ui = np.dot(self.W_, x.T) # Predict based on weights matrix
            Yi = self.activation_function(Ui) # Activation function
            error = y[i,:] - Yi.T # Error
            squared_error += 0.5*np.sum(np.power(error, 2)) # sum of squared errors
            self.W_ += self.l_rate * np.dot(error.T, x) # Weights matrix adjustment
        return squared_error

    def fit_logistic(self, X, y):
        squared_error = 0
        for i in range(X.shape[0]):
            x = np.atleast_2d(np.concatenate(([-1], X[i,:]), axis=0)) # Add bias
            Ui = np.dot(self.W_, x.T) # Predict based on weights matrix
            Yi = self.activation_function(Ui) # Activation function
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
            x = np.atleast_2d(np.concatenate(([-1], X[i,:]), axis=0)) # Add bias
            Ui = np.dot(self.W_, x.T) # Predict based on weights matrix
            Yi = self.activation_function(Ui) # Activation function
            error = y[i,:] - Yi.T # Error
            squared_error += 0.5*np.sum(np.power(error, 2)) # sum of squared errors
            derivative = 0.5*(1 - np.power(Yi, 2)) + 0.05 # sigmoid logistic derivative
            gradient = error * derivative.T # local gradient
            self.W_ += self.l_rate * np.dot(gradient.T, x) # Weights matrix adjustment
        return squared_error
        # %%% ETAPA DE TREINAMENTO
        # for t=1:Ne,
        #     Epoca=t;
        #     I=randperm(cP); P=P(:,I); T1=T1(:,I);   % Embaralha vetores de treinamento
        #     EQ=0;
        #     for tt=1:cP,   % Inicia LOOP de epocas de treinamento
        #         % CAMADA DE SAIDA
        #         X  = [-1; P(:,tt)];   % Constroi vetor de entrada com adicao da entrada x0=-1
        #         Ui = WW * X;          % Ativacao (net) dos neuronios de saida
        #         Yi = (1-exp(-Ui))./(1+exp(-Ui)); % Saida entre [-1,1]
        #         disp([Ui Yi])

        #         % CALCULO DO ERRO
        #         Ei = T1(:,tt) - Yi;           % erro entre a saida desejada e a saida da rede
        #         EQ = EQ + 0.5*sum(Ei.^2);     % soma do erro quadratico de todos os neuronios p/ VETOR DE ENTRADA

        #         %%% CALCULO DOS GRADIENTES LOCAIS
        #         Di = 0.5*(1 - Yi.^2) + 0.05;  % derivada da sigmoide logistica (camada de saida)
        #         DDi = Ei.*Di;       % gradiente local (camada de saida)

        #         % AJUSTE DOS PESOS - CAMADA DE SAIDA
        #         WW_aux=WW;
        #         WW = WW + eta*DDi*X' + mom*(WW - WW_old);
        #         WW_old=WW_aux;
        #     end   % Fim de uma epoca

        #     EQM(t)=EQ/cP;  % MEDIA DO ERRO QUADRATICO POR EPOCA
        # end   % Fim do loop de treinamento


        # %% ETAPA DE GENERALIZACAO  %%%
        # EQ2=0; HID2=[]; OUT2=[];
        # for tt=1:cQ,
        #     % CAMADA OCULTA
        #     X=[-1; Q(:,tt)];      % Constroi vetor de entrada com adicao da entrada x0=-1
        #     Ui = WW * X;          % Ativacao (net) dos neuronios da camada oculta
        #     Yi = (1-exp(-Ui))./(1+exp(-Ui));
        #     OUT2=[OUT2 Yi];       % Armazena saida da rede

        #     % CALCULO DO ERRO DE GENERALIZACAO
        #     Ei = T2(:,tt) - Yi;
        #     EQ2 = EQ2 + 0.5*sum(Ei.^2);
        # end

    def activation_function(self, Ui):
        if self.model == Models.Adaline:
            return Ui
        elif self.model == Models.Logistic:
            return (1 - np.exp(-Ui))/(1 + np.exp(-Ui))
        elif self.model == Models.MLP:
            return 1/(1 + np.exp(-Ui))

    def evaluation_nn(self, X, y):
        confusion = np.zeros((self.n_labels,self.n_labels))
        squared_error = 0
        for i in range(X.shape[0]):
            x = np.atleast_2d(np.concatenate(([-1], X[i,:]), axis=0)) # Add bias
            Ui = np.dot(x, self.W_.T) # Predict based on weights matrix
            Yi = self.activation_function(Ui) # Activation function
            squared_error += 0.5*np.sum(np.power(y[i,:] - Yi, 2)) # sum of squared errors
            predicted = np.argmax(Yi)
            real = np.argmax(y[i,:])
            confusion[predicted,real]+=1
        self.compute_confusion_matrix(confusion)
        return squared_error/X.shape[0];

    def evaluation_lmq(self, X, y):
        confusion = np.zeros((self.n_labels,self.n_labels))
        squared_error = 0
        for i in range(X.shape[0]):
            predicted = np.argmax(np.dot(X[i,:], self.W_))
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
        return

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

    def plot_learning_curve(self):
        if len(self.l_curve)>0:
            plt.plot(self.l_curve)
            plt.show()
        return
