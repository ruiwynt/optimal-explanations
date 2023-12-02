import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from ucimlrepo import fetch_ucirepo
from pmlb import fetch_data as fetch_pmlb

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(levelname)s|%(asctime)s] %(message)s", 
    datefmt="%m/%d/%Y %I:%M:%S %p"
)

UCI_DATASETS = {
    27: "credit_approval",
    94: "spambase",
    697: "academic",
    42: "glass_identification",
    15: "breast_cancer_wisconsin",
    144: "statlog",
    1: "abalone",
    186: "wine_quality",
    545: "rice",
    602: "dry_bean",
    109: "wine",
    2: "adult",
    45: "heart_disease",
    53: "iris",
    80: "handwritten_digits",
    111: "zoo",
    62: "lung_cancer",
    52: "ionosphere",
}

PMLB_DATASETS = [
    "chess",
    "connect_4",
    "contraceptive",
    "ecoli",
    "haberman",
    "labor",
    "nursery",
    "pendigits",
    "poker",
    "ring",
    "satimage",
    "schizo",
    "titanic",
    "waveform_21",
    "waveform_40",
    "yeast"
]

RANDOM_STATE = 42

class Dataset:
    def __init__(self, data, target):
        self.data = data
        self.target = target

def process_uci(dataset):
    dataset.target = dataset.target.to_numpy().flatten()
    unique = list(np.sort(np.unique(dataset.target)))
    dataset.target = np.array([unique.index(c) for c in dataset.target]).astype('int64')

    # Delete non-numerical features from data
    dataset.data = dataset.data.to_numpy()
    to_delete = []
    for i, v in enumerate(dataset.data[0]):
        if not (type(v) in (float, int) or np.isreal(v)):
            to_delete.append(i)
    for k, idx in enumerate(to_delete):
        dataset.data = np.delete(dataset.data, idx-k, axis=1)
    dataset.data = np.array(dataset.data).astype('float64')

def process_pmlb(dataset):
    unique = list(np.sort(np.unique(dataset.target)))
    dataset.target = np.array([unique.index(c) for c in dataset.target]).astype('int64')

def create_model(name, dataset):
    logging.info(f"Creating model {name}...")
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    clf = XGBClassifier(n_estimators=50, max_depth=4)
    clf.fit(X_train, y_train)
    clf.save_model(f"models/{name}.json")
    with open(f"models/{name}.lims", "w") as f:
        for (i, dmin, dmax) in zip(range(len(X)), X.min(axis=0), X.max(axis=0)):
            f.write(f"{i},{dmin},{dmax}\n")
    logging.info(f"Finished creating model {name}")

def get_datasets(cache=False):
    datasets = []
    for ds_id, ds_name in UCI_DATASETS.items():
        pkl_path = f"./datasets/{ds_name}.pkl"
        if not os.path.isfile(pkl_path):
            logging.info(f"Downloading dataset {ds_name}...")
            uci_dataset = fetch_ucirepo(id=ds_id)
            dataset = Dataset(uci_dataset.data.features, uci_dataset.data.targets)
            process_uci(dataset)
            if cache:
                with open(pkl_path, "wb") as f:
                    pickle.dump(dataset, f, protocol=pickle.DEFAULT_PROTOCOL)
                logging.info(f"Cached dataset {ds_name}")
            logging.info(f"Finished downloading dataset {ds_name}")
        else:
            with open(pkl_path, "rb") as f:
                dataset = pickle.load(f)
            logging.info(f"Loaded saved dataset {ds_name}")
        datasets.append(dataset)

    for ds_name in PMLB_DATASETS:
        pkl_path = f"./datasets/{ds_name}.pkl"
        if not os.path.isfile(pkl_path):
            logging.info(f"Downloading dataset {ds_name}...")
            X, y = fetch_pmlb(ds_name, return_X_y=True)
            dataset = Dataset(X, y)
            process_pmlb(dataset)
            if cache:
                with open(pkl_path, "wb") as f:
                    pickle.dump(dataset, f, protocol=pickle.DEFAULT_PROTOCOL)
            logging.info(f"Finished downloading dataset {ds_name}")
        else:
            with open(pkl_path, "rb") as f:
                dataset = pickle.load(f)
            logging.info(f"Loaded saved dataset {ds_name}")
        datasets.append(dataset)
    return datasets

def evaluate_model(name, dataset):
    logging.info(f"Evaluating model {name}...")
    X = dataset.data
    y = dataset.target

    clf = XGBClassifier()
    clf.load_model(f"./models/{name}.json")
    skf = StratifiedKFold(n_splits=5)
    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    logging.info(f"{name}: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    return cv_scores

def main():
    if "--cache" in sys.argv:
        datasets = get_datasets(cache=True)
    else:
        datasets = get_datasets()

    if "--create" in sys.argv:
        for name, dataset in zip(list(UCI_DATASETS.values()) + PMLB_DATASETS, datasets):
            create_model(name, dataset)
    elif "--eval" in sys.argv:
        model_scores = {}
        for name, dataset in zip(list(UCI_DATASETS.values()) + PMLB_DATASETS, datasets):
            model_scores[name] = evaluate_model(name, dataset)
        scores_df = pd.DataFrame(model_scores)
        scores_df = scores_df.agg(['mean', 'std'])
        with open("model_eval.csv", "w") as f:
            scores_df.to_csv(f, index=False)
    else:
        print("usage: python train_models.py --create | --eval [--cache]")


if __name__ == "__main__":
    # main()
    scores_df = pd.read_csv("model_eval2.csv")
    scores_df = scores_df.T
    scores_df = scores_df.reset_index()
    scores_df = scores_df.rename(columns={"index": "model_name", 0: "accuracy", 1: "std"})
    with open("model_eval2.csv", "w") as f:
        scores_df.to_csv(f, index=False)
