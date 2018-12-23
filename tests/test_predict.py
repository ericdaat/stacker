from src.core import Stacker

import pandas as pd
import numpy as np
import unittest

from sklearn import neighbors, linear_model, ensemble, svm
from sklearn.feature_extraction.text import CountVectorizer

pd.options.mode.chained_assignment = None


class TestClassification(unittest.TestCase):
    def test_predict(self):
        train = pd.read_csv('./tests/sample_data/train.csv')
        train['distance_from_center'] = (train['XCoord'] ** 2 + train['YCoord'] ** 2) ** (.5)
        v = CountVectorizer()
        train['Competitor'] = np.argmax(v.fit_transform(train['Competitor']), axis=1)

        model_array = [ensemble.RandomForestClassifier(),
                       neighbors.KNeighborsClassifier(n_neighbors=1),
                       ensemble.GradientBoostingClassifier(),
                       svm.LinearSVC()]

        stacker_model = linear_model.LogisticRegression()

        s = Stacker(model_array=model_array,
                    stacker_model=stacker_model)

        s.fit_predict(train,
                      input_features=['XCoord', 'YCoord', 'distance_from_center'],
                      labels='Competitor')


if __name__ == "__main__":
    unittest.main()
