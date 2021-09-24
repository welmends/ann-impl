from dataset import MushroomDataset, XORDataset, ORDataset
from classifier import Classifier, Models

if __name__=='__main__':
    dataset = XORDataset()
    model = Classifier(model=Models.MLP, runs=1, epochs=100, n_hidden=2, l_rate=0.1, p_train=0.8, log=True)

    model.train(dataset.X, dataset.y)
    print(model.get_stats())
    model.plot_learning_curve()
    model.plot_2d_decision_surface()
    