from dataset import MushroomDataset
from classifier import Classifier, Models

if __name__=='__main__':
    # Params
    n_runs   = 100
    epochs   = 50
    n_hidden = 2
    l_rate   = 0.01
    p_train  = 0.8

    # Dataset
    dataset  = MushroomDataset()

    # Models
    models = []
    models.append(Classifier(model=Models.Adaline, runs=n_runs, epochs=epochs, l_rate=l_rate, p_train=p_train))
    models.append(Classifier(model=Models.Logistic, runs=n_runs, epochs=epochs, l_rate=l_rate, p_train=p_train))
    models.append(Classifier(model=Models.LMQ, runs=n_runs, l_rate=l_rate, p_train=p_train))
    models.append(Classifier(model=Models.MLP, runs=n_runs, epochs=epochs, n_hidden=n_hidden, l_rate=l_rate, p_train=p_train))

    # Experiment
    for m in models:
        print(m.model.name)
        m.train(dataset.X, dataset.y)
        print(m.get_stats())
        # m.plot_learning_curve()
    