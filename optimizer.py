import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import lightgbm as lgb
max_float_digits = 4


def rounded(val):
    return '{:.{}f}'.format(val, max_float_digits)

class HyperOptTuner(object):
    """
    Tune my parameters!
    """
    def __init__(self, dtrain, dvalid, early_stopping=200, max_evals=200,
                 objective=None, metric=None, num_classes=None):
        self._counter = 0
        self._train = dtrain
        self._dvalid = dvalid
        self._early_stopping = early_stopping
        self._max_evals = max_evals
        self._tuned_params = None

        if objective is None or metric is None:
            raise ValueError(
                'You did not specify the objective or'
                'the evaluation metric. Cannot continue')

        if objective == 'multiclass':
            if num_classes is None:
                raise ValueError(
                    'objective is multiclass but num_classes has not been given'
                    'cannot continue')

        self._objective = objective
        self._num_classes = num_classes
        self._metric = metric


    def score(self, params):
        self._counter += 1
        # Edit params
        print("Iteration {}/{}".format(self._counter, self._max_evals))
        num_round = int(params['n_estimators'])
        del params['n_estimators']

        model = lgb.train(params, self._train, num_round,
                          valid_sets=[self._train, self._dvalid],
                          early_stopping_rounds=self._early_stopping,
                          verbose_eval=False)


        n_epochs = model.best_iteration
        score = list(model.best_score['valid_1'].values())[0]
        params['n_estimators'] = n_epochs
        params = dict([(key, rounded(params[key]))
                       if type(params[key]) == float
                       else (key, params[key])
                       for key in params])

        print("Trained with: ")
        print(params)
        print("\tScore {0}\n".format(score))
        return {'loss': score, 'status': STATUS_OK, 'params': params}

    def optimize(self, trials):
        space = {
            'n_estimators': 10000,  # hp.quniform('n_estimators', 10, 1000, 10),
            'num_leaves': hp.choice('num_leaves', np.arange(10, 200, dtype=int)),
            'min_child_weight': hp.quniform('min_child_weight', 0.01, 10, 0.5),
            'subsample': hp.uniform('subsample', 0.4, 1),
            'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.4, 1, 0.25),
            'metric': self._metric,
            'objective': self._objective,
        }

        if self._num_classes is not None:
            space['num_classes'] = self._num_classes

        fmin(self.score, space, algo=tpe.suggest, trials=trials, max_evals=self._max_evals),

        min_loss = 1
        min_params = {}
        for trial in trials.trials:
            tmp_loss, tmp_params = trial['result']['loss'], trial['result']['params']
            if tmp_loss < min_loss:
                min_loss, min_params = tmp_loss, tmp_params

        print("Winning params:")
        print(min_params)
        print("\tScore: {}".format(1-min_loss))
        self._tuned_params = min_params

    def tune(self):
        print("Tuning...\n")
        # Trials object where the history of search will be stored
        trials = Trials()
        self.optimize(trials)
