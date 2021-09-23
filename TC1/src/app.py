from dataset import MushroomDataset, XORDataset, ORDataset
from classifier import Classifier, Models

if __name__=='__main__':
    dataset = MushroomDataset()
    model = Classifier(model=Models.MLP)

    model.train(dataset.X, dataset.y)
    print(model.get_stats())
    model.plot_learning_curve()
    model.plot_2d_decision_surface()
    