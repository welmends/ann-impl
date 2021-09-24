from dataset import MushroomDataset
from classifier import Classifier, Models

if __name__=='__main__':
    # Params
    n_runs   = 1
    epochs   = 3
    n_hidden = 10
    l_rate   = 0.01
    p_train  = 0.8
    log      = True

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
        method.train(dataset.X, dataset.y)
        method.save_stats()
        method.save_learning_curve()
        method.save_all_learning_curves()
    