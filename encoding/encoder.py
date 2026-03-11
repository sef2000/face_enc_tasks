import numpy as np
from sklearn.linear_model import RidgeCV
# import pls
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
from tqdm import tqdm

class EncodingRidge:
    def __init__(self, scoring="r2", cv=5):
        self.scoring = scoring
        self.cv = cv
        #self.model = RidgeCV(scoring=scoring, cv=5, fit_intercept=False)
        self.model = PLSRegression(n_components=4, scale=False) # as in her paper https://www.nature.com/articles/s41562-025-02218-1#Abs1
        self.scale = StandardScaler()

    def fit(self, x_train, y_train, x_test, y_test):
        # scale
        x_train = self.scale.fit_transform(x_train)
        x_test = self.scale.transform(x_test)
        y_train = self.scale.fit_transform(y_train)
        y_test = self.scale.transform(y_test)

        self.model.fit(x_train, y_train)

        if self.scoring == "explained_variance":
            score = explained_variance_score(y_test, self.model.predict(x_test))
        else:
            score = self.model.score(x_test, y_test)

        return score

    def folds(self, x, y):
        """
        MCCV, but keeping splits constant between targets and models, to get more stable estimates of scores. Still seperate
        ridge regressions are used per target and model to sufficiently tune alpha.
        :param x: shape (models, trials, features)
        :param y: shape (trials, targets)
        :return: shape (cv, models, targets)
        """
        ss = ShuffleSplit(n_splits=self.cv, test_size=0.5)
        scores_splits = []
        for train_index, test_index in tqdm(ss.split(x[0])):
            y_train, y_test = y[train_index], y[test_index]
            scores_model = []
            for model_id in range(x.shape[0]):
                x_train, x_test = x[model_id][train_index], x[model_id][test_index]
                scores_neuron = []
                for neuron in range(y.shape[1]):
                    print(f"Model {model_id}, Neuron {neuron}")
                    print(f"Train shape: {x_train.shape}, {y_train[:, neuron].reshape(-1,1).shape}")
                    score = self.fit(x_train, y_train[:, neuron].reshape(-1,1), x_test, y_test[:, neuron].reshape(-1,1))
                    scores_neuron.append(score)
                scores_model.append(scores_neuron)
            scores_splits.append(scores_model)
        return np.array(scores_splits)

if __name__ == "__main__":
    # test with random data
    x = np.random.rand(6, 300, 7) # models, trials, features
    y = np.random.rand(300, 20) # trials, targets

    encoding = EncodingRidge(scoring="explained_variance", cv=5)
    scores = encoding.folds(x, y)
    print(scores.shape) # should be (cv, models, targets)