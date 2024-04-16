import numpy as np
import pandas as pd


class MyKNNClf:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.train_size = None
        self.X, self.y = None, None
        self.size = None
        self.metric = metric

    def __str__(self):
        return self.train_size

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.train_size = X.shape
        self.size = self.X.shape[0]

    def predict(self, X):
        result = X.apply(self.calculating_the_distance, axis=1)
        return result

    def predict_proba(self, X):
        result = X.apply(self.proba_calculating_the_distance, axis=1)
        return np.array(list(result))

    def calculating_the_distance(self, value):
        data = {f'col{i}': val for i, val in enumerate(list(value))}
        X = pd.DataFrame(data, index=[0])
        X = pd.concat([X] * self.size, ignore_index=True)
        X.columns = list(self.X.columns)
        if self.metric == 'euclidean':
            evD = pd.DataFrame(np.sqrt(((self.X.reset_index(drop=True) - X) ** 2).sum(axis=1)))
        elif self.metric == 'chebyshev':
            evD = pd.DataFrame((self.X.reset_index(drop=True) - X).abs().max(axis=1))
            evD = evD.reset_index(drop=True)
        elif self.metric == 'manhattan':
            evD = pd.DataFrame((self.X.reset_index(drop=True) - X).abs().sum(axis=1))
            evD = evD.reset_index(drop=True)
        else:
            evD = pd.DataFrame(1 - ((self.X.reset_index(drop=True) * X).sum(axis=1) / (
                        np.sqrt((X ** 2).sum(axis=1)) * np.sqrt((self.X.reset_index(drop=True) ** 2).sum(axis=1)))))
            evD = evD.reset_index(drop=True)
        evD.columns = ['bD']
        evD['y'] = self.y.reset_index(drop=True)
        result = evD.sort_values(by='bD').reset_index(drop=True).iloc[:self.k]['y'].mode()
        if len(result) == 1:
            return result[0]
        return 1

    def proba_calculating_the_distance(self, value):
        data = {f'col{i}': val for i, val in enumerate(list(value))}
        X = pd.DataFrame(data, index=[0])
        X = pd.concat([X] * self.size, ignore_index=True)
        X.columns = list(self.X.columns)
        if self.metric == 'euclidean':
            evD = pd.DataFrame(np.sqrt(((self.X.reset_index(drop=True) - X) ** 2).sum(axis=1)))
        elif self.metric == 'chebyshev':
            evD = pd.DataFrame((self.X.reset_index(drop=True) - X).abs().max(axis=1))
            evD = evD.reset_index(drop=True)
        elif self.metric == 'manhattan':
            evD = pd.DataFrame((self.X.reset_index(drop=True) - X).abs().sum(axis=1))
            evD = evD.reset_index(drop=True)
        else:
            evD = pd.DataFrame(1 - ((self.X.reset_index(drop=True) * X).sum(axis=1) / (
                        np.sqrt((X ** 2).sum(axis=1)) * np.sqrt((self.X.reset_index(drop=True) ** 2).sum(axis=1)))))
            evD = evD.reset_index(drop=True)
        evD.columns = ['bD']
        evD['y'] = self.y.reset_index(drop=True)
        result = evD.sort_values(by='bD').reset_index(drop=True).iloc[:self.k]['y']
        return list(result).count(1) / len(list(result))
