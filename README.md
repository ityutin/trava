[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/) [![CodeFactor](https://www.codefactor.io/repository/github/ityutin/trava/badge)](https://www.codefactor.io/repository/github/ityutin/trava) <a href="https://codeclimate.com/github/ityutin/trava/maintainability"><img src="https://api.codeclimate.com/v1/badges/7198080c24ab23f3e113/maintainability" /></a> [![codecov](https://codecov.io/gh/ityutin/trava/branch/master/graph/badge.svg)](https://codecov.io/gh/ityutin/trava)

# 🌿 Trava ( initially stands for TrainValidation )
Framework that helps to train models, compare them and track parameters&metrics along the way. Works with tabular data only.

```
pip install trava
```

# Compare models and keep track of metrics with ease!

While working on a project, we often experiment with different models looking at the same metrics. 
For example, we log those that can be represented as a single number, however some of them require graphs to make sense. It's also useful to save those metrics somewhere for future analysis, the list can go on.
## So why not to use some unified interface for that? 
## Here is Trava's way:

## 1). Declare metrics you want to calculate:

```python
# in this case, sk and sk_proba are just wrappers around sklearn's metrics
# but you can use any metric implementation you want
scorers = [
  sk_proba(log_loss),
  sk_proba(roc_auc_score),
  sk(recall_score),
  sk(precision_score),
]
```

## 2). What do you want to do with the metrics?

```python
# let's log the metrics
logger_handler = LoggerHandler(scorers=scorers)
```

## 3). Initialize Trava

```python
trava = TravaSV(results_handlers=[logger_handler])
```

## 4). Fit your model using Trava

```python
# prepare your data
X_train, X_test, y_train, y_test = ...

split_result = SplitResult(X_train=X_train, 
                           y_train=y_train,
                           X_test=X_test,
                           y_test=y_test)

trava.fit_predict(raw_split_data=split_result,
                  model_type=GaussianNB, # we pass model class and parameters separately
                  model_init_params={},  # to be able to track them properly
                  model_id='gnb') # just a unique identifier for this model
```

**fit_predict** call does roughly the same as:

```python
gnb = GaussianNB()
gnb.fit(split_result.X_train, split_result.y_train)
gnb.predict(split_result.X_test)
```

But now you don't care how the metrics you declared are calculated. You just get them in your console! Btw, those metrics definitely need to be improved. :]

```bash
Model evaluation nb
* Results for gnb model *
Train metrics:
log_loss:
16.755867191506482
roc_auc_score:
0.7746522424770221
recall_score:
0.10468384074941452
precision_score:
0.9122448979591836


Test metrics:
log_loss:
16.94514025416013
roc_auc_score:
0.829444814485889
recall_score:
0.026041666666666668
precision_score:
0.7692307692307693
```

After training multiple models you can get all metrics for all models by calling.

```python
trava.results
```

Get the full picture and more examples by going through [the guide notebook!](https://github.com/ityutin/trava/blob/master/examples/Basics.ipynb)

### Built-in handlers:
- **LoggerHandler** - logs metrics
- **PlotHandler** - plots metrics
- **MetricsDictHandler** - returns all metrics wrapped in a dict

# Enable metrics autotracking. How cool is that?
Experiments tracking is a must in Data Science, so you shouldn't neglect that. You may integrate any tracking framework in **Trava**! **Trava** comes with **MLFlow** tracker ready-to-go.
It can autotrack:
- model's parameters
- any metric
- plots
- serialized models

## MLFlow example:

```python
# get tracker's instance
tracker = MLFlowTracker(scorers=scorers)
# initialize Trava with it
trava = TravaSV(tracker=tracker)
# fit your model as before
trava.fit_predict(raw_split_data=split_result,
                  model_type=GaussianNB,
                  model_id='gnb')
```

Done. All model parameters and metrics are now tracked!
Also supported tracking of:
- cross-validation case with nested tracking
- eval results for common boosting libraries ( **XGBoost**, **LightGBM**, **CatBoost** )

Checkout a detailed notebooks how to [track metrics & parameters](https://github.com/ityutin/trava/blob/master/examples/MLFlow_basic.ipynb) and [plots & serialized models](https://github.com/ityutin/trava/blob/master/examples/MLFlow_advanced.ipynb).

# General information

- highly customizable training & evaluation processes ( see **trava.fit_predictor.py.FitPredictor** class and its subclasses )
- built-in train/test/validation split logic
- common boosting libraries extensions ( for early-stopping with validation sets )
- tracks model parameters, metrics, plots, serialized models. it's easy to integrate any tracking framework of your choice
- you are also able to evaluate metrics after **fit_predict** call, if you forgot to add some metric
- you are able to evaluate metrics even when your data and even a trained model are already unloaded ( depends on a metric used, true most of the times )
- now only supervised learning problems are supported yet there is a potential to extend it to support unsupervised problems
- unit-tested
- I use it every day for my needs thus I care about the quality and reliability

Only sklearn-style model are supported for the time being. ( it uses **fit**, **predict**, **predict_proba** methods )


### Requirements

```
pandas
numpy
python 3.7
``` 

It's also convenient to use the lib with [sklearn](https://github.com/scikit-learn/scikit-learn) ( e.g. for taking metrics from there. ). Also couple of extensions are based on sklearn classes.
