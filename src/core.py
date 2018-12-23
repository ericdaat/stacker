import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score


class Stacker(object):
    """The Stacker class
    """
    def __init__(self, model_array, stacker_model, n_folds=5):
        """ Instantiates the Stacker class.

        Args:
            model_array (list): list of sklearn models
            stacker_model: meta model that will give the final prediction based on the output
                of the other models.
            n_folds (int): number of folds that we divide the dataset into. A model will learn
                on n-1 folds and predict on the other one.
        """
        self._model_array = model_array
        self._stacker_model = stacker_model
        self._n_folds = n_folds

    def _fill_df_with_fold_ids(self, df):
        """ Takes a pandas DataFrame and add fold ids to it.

        Args:
            df (pd.DataFrame): Dataframe that holds the data

        Returns (pd.DataFrame): the dataframe with folds added

        """
        df['fold_id'] = np.random.permutation((df.index % self._n_folds + 1).tolist())

        return df

    def _make_model_name(self, model):
        """ Sets a name on a model

        Args:
            model: sklearn model

        Returns (str): The name of the model

        """
        return 'M_{0}'.format(type(model).__name__)

    def _make_meta_df(self, df):
        """ Make a meta DataFrame, that is ready for training multiple models.

        Args:
            df (pd.DataFrame): Input raw dataframe.

        Returns (pd.DataFrame): The input dataframe with folds and NaN predictions.

        """
        meta = df.copy()
        meta = self._fill_df_with_fold_ids(meta)

        for model_index, model in enumerate(self._model_array):
            meta[self._make_model_name(model)] = np.nan

        return meta

    def _fill_train_meta_with_predictions(self, meta_df, input_features, labels):
        """ Make every model output their predictions on the meta dataframe.
        Note that the final prediction is not yet given.

        Args:
            meta_df (pd.DataFrame): the input meta dataframe
            input_features (list): list of input features, corresponding to column names
                from the meta dataframe.
            labels (str): name of the column containing the labels to predict

        Returns (pd.DataFrame): DataFrame with predictions from every model.
        """
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

    def _cross_val_train(self, filled_meta_df, input_features, labels):
        """ Make the final prediction based on the output of every model, using
            cross validation, returning the mean accuracy.

        Args:
            filled_meta_df (pd.DataFrame): The meta dataframe, filled with predictions
                from every model but the final one.
            input_features (list): List of features, corresponding to column names from
                the meta dataframe.
            labels (str): name of the column containing the labels to predict

        Returns (int): Mean accuracy
        """
        X = np.hstack([filled_meta_df[input_features],
                       pd.concat([pd.get_dummies(filled_meta_df[self._make_model_name(model)])
                                  for model in self._model_array], axis=1)])

        y = filled_meta_df[labels]

        cv_score = cross_val_score(self._stacker_model, X, y,
                                   cv=5, scoring='accuracy')

        return np.mean(cv_score)

    def fit_predict(self, df, input_features, labels):
        """ Fit and predict, runs the whole pipeline, from data preparation, to the final
        prediction based on the output of every model to be stacked.

        Args:
            df (pd.DataFrame): Input dataframe
            input_features (list): List of features, corresponding to column names from
                the dataframe.
            labels (str): name of the column containing the labels to predict

        Returns (int): Mean accuracy on training dataset, using cross validation
        """
        train_meta = self._make_meta_df(df)
        train_meta = self._fill_train_meta_with_predictions(train_meta,
                                                            input_features=input_features,
                                                            labels=labels)
        return self._cross_val_train(train_meta,
                                     input_features=input_features,
                                     labels=labels)
