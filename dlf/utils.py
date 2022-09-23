import numpy as np


def precision(y_pred, y_true):
    assert len(y_pred.shape) == 1 and len(y_true.shape) == 1,\
        "y_pred and y_true must be vector of classes"
    assert y_pred.size == y_true.size,\
        "y_pred and y_true must have the same size"
    classes = np.unique(y_true)
    precisions = []
    for i in classes:
        tp = np.logical_and((y_pred==i), (y_true==i)).sum()
        fp = np.logical_and((y_pred==i), (y_true!=i)).sum()
        precisions.append(tp/(tp+fp))
    return np.array(precisions)


def recall(y_pred, y_true):
    assert len(y_pred.shape) == 1 and len(y_true.shape) == 1,\
        "y_pred and y_true must be vector of classes"
    assert y_pred.size == y_true.size,\
        "y_pred and y_true must have the same size"
    classes = np.unique(y_true)
    recalls = []
    for i in classes:
        tp = np.logical_and((y_pred==i), (y_true==i)).sum()
        fn = np.logical_and((y_pred!=i), (y_true==i)).sum()
        recalls.append(tp/(tp+fn))
    return np.array(recalls)


def f_score(y_pred, y_true, beta=1):
    prec = precision(y_pred, y_true)
    rec = recall(y_pred, y_true)
    beta_square = beta**2
    return ((1+beta_square)*prec*rec)/(prec+rec*beta_square)


def confusion_matrix(y_pred, y_true):
    assert len(y_pred.shape) == 1 and len(y_true.shape) == 1,\
        "y_pred and y_true must be vector of classes"
    assert y_pred.size == y_true.size,\
        "y_pred and y_true must have the same size"
    classes = np.unique(y_true)
    classes.sort()
    conf_m = np.zeros(shape=(len(classes), len(classes)))
    for i in classes:
        for j in classes:
            conf_m[i, j] = np.logical_and((y_pred==i), (y_true==j)).sum()
    return conf_m, classes


def accuracy(y_pred, y_true):
    assert len(y_pred.shape) == 1 and len(y_true.shape) == 1,\
        "y_pred and y_true must be vector of classes"
    assert y_pred.size == y_true.size,\
        "y_pred and y_true must have the same size"
    return (y_pred==y_true).sum()/y_true.size