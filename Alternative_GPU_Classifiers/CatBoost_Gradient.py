import catboost as cb
import catboost.datasets as cbd
import catboost.utils as cbu
import numpy as np
import pandas as pd
import tensorflow as tf  # Just for checking if GPU is available :)
from openBCI import config as cf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Checking if GPU is available
GPU_AVAILABLE = tf.test.is_gpu_available()
print("GPU available:", GPU_AVAILABLE)

'''
data = pd.read_csv(cf.prepared_data_15min)
data_tr = data.loc[:130000]
data_ts = data.loc[130001:]
print(data.shape)
StdScaler = StandardScaler()
X_Train = data_tr.drop(['0'], axis=1)
X_Train = StdScaler.fit_transform(X_Train)
Y_Train = data_tr[['0']].values.ravel()

x_test = data_ts.drop(['0'], axis=1)
x_test = StdScaler.fit_transform(x_test)
y_test = data_ts[['0']].values.ravel()
'''

# Get csv data
data = pd.read_csv(cf.prepared_data_15min)

X = data.drop(['0'], axis=1)
y = data[['0']].values.ravel()

# p_data = y.value_counts()
#print(p_data)

# Feature Scaling
StdScaler = StandardScaler()
X_scaled = StdScaler.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_Train, x_test, Y_Train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=0)


estimator = cb.CatBoostClassifier(depth=10, iterations=600, verbose=10, task_type='GPU', learning_rate=0.08, allow_writing_files=False, loss_function='MultiClass')
#estimator = cb.CatBoostClassifier(l2_leaf_reg=0.1, depth=8, iterations=1550, verbose=10, task_type='GPU', learning_rate=0.0775, allow_writing_files=False)


estimator.fit(X_Train, Y_Train)
pred = estimator.predict(x_test)

print("Saving model...")
estimator.save_model('../models/CatBoost.mlmodel')
ac = accuracy_score(y_test, pred)

print(ac)


'''
class UciAdultClassifierObjective(object):
    def __init__(self, dataset, const_params, fold_count):
        self._dataset = dataset
        self._const_params = const_params.copy()
        self._fold_count = fold_count
        self._evaluated_count = 0

    def _to_catboost_params(self, hyper_params):
        return {
            'learning_rate': hyper_params['learning_rate'],
            'depth': hyper_params['depth'],
            'l2_leaf_reg': hyper_params['l2_leaf_reg']}

    # hyperopt optimizes an objective using `__call__` method (e.g. by doing
    # `foo(hyper_params)`), so we provide one
    def __call__(self, hyper_params):
        # join hyper-parameters provided by hyperopt with hyper-parameters
        # provided by the user
        params = self._to_catboost_params(hyper_params)
        params.update(self._const_params)

        print('evaluating params={}'.format(params), file=sys.stdout)
        sys.stdout.flush()

        # we use cross-validation for objective evaluation, to avoid overfitting
        scores = cb.cv(
            pool=self._dataset,
            params=params,
            fold_count=self._fold_count,
            partition_random_seed=20181224,
            verbose=False)

        # scores returns a dictionary with mean and std (per-fold) of metric
        # value for each cv iteration, we choose minimal value of objective
        # mean (though it will be better to choose minimal value among all folds)
        # because noise is additive
        max_mean_auc = np.max(scores['test-AUC-mean'])
        print('evaluated score={}'.format(max_mean_auc), file=sys.stdout)

        self._evaluated_count += 1
        print('evaluated {} times'.format(self._evaluated_count), file=sys.stdout)

        # negate because hyperopt minimizes the objective
        return {'loss': -max_mean_auc, 'status': hyperopt.STATUS_OK}


def find_best_hyper_params(dataset, const_params, max_evals=100):
    # we are going to optimize these three parameters, though there are a lot more of them (see CatBoost docs)
    parameter_space = {
        'learning_rate': hyperopt.hp.uniform('learning_rate', 0.2, 1.0),
        'depth': hyperopt.hp.randint('depth', 7),
        'l2_leaf_reg': hyperopt.hp.uniform('l2_leaf_reg', 1, 10)}
    objective = UciAdultClassifierObjective(dataset=dataset, const_params=const_params, fold_count=6)
    trials = hyperopt.Trials()
    best = hyperopt.fmin(
        fn=objective,
        space=parameter_space,
        algo=hyperopt.rand.suggest,
        max_evals=max_evals,
        rstate=np.random.RandomState(seed=20181224))
    return best


def train_best_model(X, y, const_params, max_evals=100, use_default=False):
    # convert pandas.DataFrame to catboost.Pool to avoid converting it on each
    # iteration of hyper-parameters optimization
    dataset = cb.Pool(X, y) # cat_features=np.where(X.dtypes != np.float)[0])

    if use_default:
        # pretrained optimal parameters
        best = {
            'learning_rate': 0.4234185321620083,
            'depth': 5,
            'l2_leaf_reg': 9.464266235679002}
    else:
        best = find_best_hyper_params(dataset, const_params, max_evals=max_evals)

        # merge subset of hyper-parameters provided by hyperopt with hyper-parameters
        # provided by the user
        hyper_params = best.copy()
        hyper_params.update(const_params)

        # drop `use_best_model` because we are going to use entire dataset for
        # training of the final model
        hyper_params.pop('use_best_model', None)

        model = cb.CatBoostClassifier(**hyper_params)
        model.fit(dataset, verbose=False)

        return model, hyper_params

have_gpu = True
# skip hyper-parameter optimization and just use provided optimal parameters
use_optimal_pretrained_params = True
# number of iterations of hyper-parameter search
hyperopt_iterations = 30

const_params = dict({
    'task_type': 'GPU' if have_gpu else 'CPU',
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
    #'custom_metric': ['AUC'],
    'iterations': 100,
    'random_seed': 20181224})

model, params = train_best_model(
    X_Train, Y_Train,
    const_params,
    max_evals=hyperopt_iterations)
    #use_default=use_optimal_pretrained_params)
print('best params are {}'.format(params), file=sys.stdout)


def calculate_score_on_dataset_and_show_graph(X, y, model):
    import sklearn.metrics
    import matplotlib.pylab as pl
    pl.style.use('ggplot')

    dataset = cb.Pool(X, y, cat_features=np.where(X.dtypes != np.float)[0])
    fpr, tpr, _ = cbu.get_roc_curve(model, dataset, plot=True)
    auc = sklearn.metrics.auc(fpr, tpr)
    return auc

calculate_score_on_dataset_and_show_graph(x_test, y_test, model)
'''