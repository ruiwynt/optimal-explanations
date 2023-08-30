import json


def get_xgboost_thresholds(filename: str):
    with open(filename, "r") as fd:
        model = json.load(fd)
    json_trees = model["learner"]["gradient_booster"]["model"]["trees"]

    thresholds = {}
    for tree in json_trees:
        for f_id, threshold in zip(tree["split_indices"], tree["split_conditions"]):
            if not f_id in thresholds.keys():
                thresholds[f_id] = set([threshold])
            else:
                thresholds[f_id].add(threshold)
    for f_id in thresholds.keys():
        thresholds[f_id] = sorted(list(thresholds[f_id]))
    return thresholds


if __name__ == "__main__":
    thresholds = get_xgboost_thresholds("model.json")
    print(json.dumps(thresholds, indent=2))
