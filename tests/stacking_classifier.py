from stacking.core import Stacker

import pandas as pd
import numpy as np

from sklearn import neighbors
from sklearn import linear_model
from sklearn import ensemble
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer

pd.options.mode.chained_assignment = None


if __name__ == '__main__':
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
