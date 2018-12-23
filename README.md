# Sklearn Model Stacker

## About

This is a Python package that aims at building a high level API for 
easily stacking several scikit-learn models, hopefully resulting in more accurate
predictions.

It is inspired by the blog post [Guide to Model Stacking (i.e. Meta Ensembling)](https://gormanalysis.com/guide-to-model-stacking-i-e-meta-ensembling/),
written by Ben Gorman.


## Installation

Create a virtual environment, and install the requirements.
```
virtualenv venv -p python3;
source venv/bin/activate;
pip install -r requirements.txt;
```


## Example

> Some players are playing at throwing darts, and we are given a dataset
gathering the coordinates of the landed darts, as well as who threw them.  
Build a model that predicts who threw the dart based on the observed coordinates.

The idea here is to use stacking, to combine multiple classification models.

The training dataset might look like this:

| ID | XCoord | YCoord | distance_from_center | Competitor |
|----|--------|--------|----------------------|------------|
| 0  | 0.06   | 0.36   | 0.36                 | 1          |
| 1  | -0.77  | -0.26  | 0.81                 | 2          |


And the code to make predictions using stacking would look like that:
``` python
from src.core import Stacker

from sklearn import neighbors, linear_model, ensemble, svm
from sklearn.feature_extraction.text import CountVectorizer

train = pd.DataFrame([
    {"XCoord": 0.06, "YCoord": 0.36, "distance_from_center": 0.36, "Competitor": 1},
    {"XCoord": -0.77, "YCoord": -0.26, "distance_from_center": 0.81, "Competitor": 2}
]

# Models that are going to be stacked
model_array = [ensemble.RandomForestClassifier(),
               neighbors.KNeighborsClassifier(n_neighbors=1),
               ensemble.GradientBoostingClassifier(),
               svm.LinearSVC()]

# Model that will output the final prediction based on the other models input
stacker_model = linear_model.LogisticRegression()

# Build the stacker object
s = Stacker(model_array=model_array,
            stacker_model=stacker_model)

# Fit and predict
mean_accuracy = s.fit_predict(train,
                              input_features=["XCoord", "YCoord", "distance_from_center"],
                              labels="Competitor")
```

