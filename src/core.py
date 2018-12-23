import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score


class Stacker(object):
    def __init__(self, model_array, stacker_model, n_folds=5):
        self._model_array = model_array
        self._stacker_model = stacker_model
        self._n_folds = n_folds

    def _fill_df_with_fold_ids(self, df):
        df['fold_id'] = np.random.permutation((df.index % self._n_folds + 1).tolist())

        return df

    def _make_model_name(self, model):
        return 'M_{0}'.format(type(model).__name__)

    def make_meta_df(self, df):
        meta = df.copy()
        meta = self._fill_df_with_fold_ids(meta)

        for model_index, model in enumerate(self._model_array):
            meta[self._make_model_name(model)] = np.nan

        return meta

    def fill_train_meta_with_predictions(self, meta_df, input_features, labels):
        folds_with_predictions = []

        for i in range(1, self._n_folds):
            train_fold = meta_df.loc[meta_df['fold_id'] != i]
            test_fold = meta_df.loc[meta_df['fold_id'] == i]

            X_train = train_fold[input_features]
            y_train = train_fold[labels]
            X_test = test_fold[input_features]
            y_test = test_fold[labels]

            for model in self._model_array:
                model.fit(X_train, y_train)
                test_fold[self._make_model_name(model)] = model.predict(X_test)
                folds_with_predictions.append(test_fold)

        return pd.concat(folds_with_predictions)

    def cross_val_train(self, filled_meta_df, input_features, labels):
        X = np.hstack([filled_meta_df[input_features],
                       pd.concat([pd.get_dummies(filled_meta_df[self._make_model_name(model)])
                                  for model in self._model_array], axis=1)])

        y = filled_meta_df[labels]

        cv_score = cross_val_score(self._stacker_model, X, y,
                                   cv=5, scoring='accuracy')

        return np.mean(cv_score)

    def fit_predict(self, df, input_features, labels):
        train_meta = self.make_meta_df(df)
        train_meta = self.fill_train_meta_with_predictions(train_meta,
                                                           input_features=input_features,
                                                           labels=labels)
        return self.cross_val_train(train_meta,
                                    input_features=input_features,
                                    labels=labels)
