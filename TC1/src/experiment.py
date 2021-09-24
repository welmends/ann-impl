from dataset import MushroomDataset
from classifier import Classifier, Models

if __name__=='__main__':
    # Params
    n_runs   = 100
    epochs   = 50
    n_hidden = 2
    l_rate   = 0.01
    p_train  = 0.8
    log      = False

    # Dataset
    dataset  = MushroomDataset()

    # Models
    models = [Models.Adaline, Models.Logistic, Models.LMQ, Models.MLP]
    methods = []
    for model in models:
        c = Classifier(model=model, runs=n_runs, epochs=epochs, n_hidden=n_hidden, l_rate=l_rate, p_train=p_train, log=log)
        methods.append(c)

    # Experiment
    for method in methods:
        print(method.model.name)
        method.train(dataset.X, dataset.y)
        print(method.get_stats())
    