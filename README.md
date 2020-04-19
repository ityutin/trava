# 🌿 Trava ( initially stands for TrainValidation )
Framework that helps to train models, compare them and track parameters&metrics along the way. Works with tabular data only.

## Why

When experimenting with some data&models, notebooks quickly become messy and unreliable. Usually when we solve some problem we are focused on some set of metrics and we want to compare models with each other. This lib tries to provide unified interface for this and other tasks.

Another important thing is experiment tracking. **Trava** helps you to track all the model parameters as well as metrics. You may subclass TravaTracker to support you tracking system. Now **Trava** goes with ready-to-go MLFlowTracker. 

## How

You tell what metrics you want to calculate and how results should be presented to you. Then you just run **Trava** using a model of your choice and parameters for it. Fit&predict process is customizable as well. See examples/ dir for the details. For now only sklearn-style model are supported. ( *fit*, *predict*, *predict_proba* methods )

## Example

Note: See examples/Basics.ipynb for the intro tour.


```
# what metrics to calculate. sk(...) means wrapper for sklearn metrics, custom metrics are easily supported as well.
scorers = [sk(recall_score), sk(precision_score)]

# how to show the metrics. In this case dictionary with metrics values will be returned
dict_handler = MetricsDictHandler(scorers=output_scorers)

# prepare data
df = pd.read_csv('...')

split_config = DataSplitConfig(split_logic=BasicSplitLogic(shuffle=True),
                               target_col_name='target',
                               test_size=0.3)
# just splits data into Train/Test
split_result = Splitter.split(df=df, config=split_config)

# initialize Trava
trava = TravaSV(results_handlers=[dict_handler])

# get your results
trava.fit_predict(raw_split_data=split_result, 
                  model_id='xgb',  # uniquely identifies your model
                  model_type=xgb.XGBClassifier,  # what model to run
                  model_init_params={'max_depth': 3})  # parameters to init model with

# then go on playing with other models 
...
# call this to get all previous results at once
trava.results 
```

### Prerequisites

```
pandas
numpy
python 3.7 ( the true minimum version is not yet confirmed ) 
``` 

The lib was written using Python 3.7, yet I currently don't know the minimum Python version required.  

It's also convenient to use the lib with sklearn ( e.g. for taking metrics from there. ). Also couple of extensions are based on sklearn classes.
