from datetime import datetime as _dt

import numpy as _np
from sklearn.metrics import log_loss as _log_loss
from sklearn.model_selection import StratifiedKFold as _StratifiedKFold

__all__ = ['cross_val_predict_proba']


def cross_val_predict_proba(classifier, X, y, cv=3, random_state=None, verbose=0):
    """
    Parameters
    ----------
    classifier
    X: _np.ndarray
    y: _np.ndarray
    cv: int or _StratifiedKFold
    random_state: int
    verbose: int

    Returns
    -------

    """
    cv = _StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state) if type(cv) is int else cv
    cross_scores = _np.empty(0)
    predict_proba = _np.empty((y.shape[0], len(_np.unique(y))))
    for i, (index_train, index_test) in enumerate(cv.split(X, y), 1):
        start_time = _dt.now()
        if verbose > 0:
            print('cv=%d' % i, end=' ', flush=True)

        classifier.fit(X[index_train], y[index_train])
        try:
            predict_proba[index_test] = classifier.predict_proba(X[index_test])
        except AttributeError:
            predict_proba[index_test] = classifier.predict(X[index_test])
        score = _log_loss(y[index_test], predict_proba[index_test])
        cross_scores = _np.append(cross_scores, [score])

        if verbose > 0:
            print('score=%f' % score, 'elapsed time:', _dt.now() - start_time)
    return predict_proba, cross_scores
