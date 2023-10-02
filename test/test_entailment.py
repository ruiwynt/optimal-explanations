import json
import numpy as np
from xgboost import XGBClassifier, DMatrix
from sklearn import datasets

from src.entailment import EntailmentChecker
from src.model import Model

def test_iris():
    path = "examples/iris.json"
    data = datasets.load_iris()
    _verify_predictions(path, data)

def test_wine():
    path = "examples/wine.json"
    data = datasets.load_wine()
    _verify_predictions(path, data)

def test_breast_cancer():
    path = "examples/breast_cancer.json"
    data = datasets.load_breast_cancer()
    _verify_predictions(path, data)

def _verify_predictions(path, data):
    with open(path, "r") as f:
        model = Model(json.loads(f.read()))
    entailer = EntailmentChecker(model)
    xgb = XGBClassifier()
    xgb.load_model(path)

    X = data.data
    y = data.target

    xgb_ws = np.array([x for x in xgb.get_booster().predict(DMatrix(X))])
    xgb_preds = np.array([x for x in xgb.predict(X)])
    ent_ws = np.array([entailer._get_weights(x) for x in X])
    ent_preds = np.array([entailer.predict([0], ws=ws) for ws in ent_ws])

    if len(ent_ws[0]) == 1:
        # binary::logistic objective
        ent_ws = ent_ws.flatten()
        ent_ws = 1/(1+np.exp(-ent_ws))
    else:
        # multi::softprob objective
        ent_ws = np.exp(ent_ws)
        ent_ws = ent_ws/np.sum(ent_ws, axis=1, keepdims=True)

    n = len(xgb_preds)
    nw = n if len(xgb_ws.shape) == 1 else n * xgb_ws.shape[1]
    x_wrong = np.sum(xgb_preds != ent_preds)
    w_wrong = np.sum(np.logical_not(np.isclose(xgb_ws, ent_ws)))
    assert x_wrong == 0
    assert w_wrong == 0
    print(f"WRONG WEIGHTS: {w_wrong}/{nw} | {round(100*w_wrong/nw, 2)}%")
    print(f"WRONG PREDICTIONS: {x_wrong}/{n} | {round(100*x_wrong/n, 2)}%")