# xregion - Optimally Informative and Robust Explanations for Boosted Trees

## 1. Introduction

xregion is a tool which computes an optimal explanation for a prediction for a Boosted Tree. More specifically, given an input instance x with predicted class c, output an explanation in the form of a set of intervals on each input feature such that any other instance y which has feature values within the intervals of the explanation is always classified as c. This is the implementation of the procedures outlined in the Honours thesis. 

Currently, xregions supports XGBoost trees trained with the `binary::logistic` and `multi::softprob` objectives. Categorical features are not supported.

## 2. Usage

First, install all requirements. A working installation of the Z3 theorem prover is required.

Save the desired XGBoost Tree as a .json file using the `save_model` method of the classifier and put it into the `./models` folder. To define the minimum and maximum values of each feature's domain, create a .lims file with the same name as the model.json file and put it into the models folder. Each line of the .lims file should have the format `<feature_id>, <domain_min>, <domain_max>`. The `feature_id` field should start from zero and be sequential with no numbers missing.

To compute an optimal explanation for an instance, use the command:

```
python xregions.py -m <model_name> -e ([v1, v2, ..., vm]|random) --seed-gen <seed_generation_method>
```

where v1, v2, ... etc are the feature values of the input instance. To compute the optimal explanation for an arbitrary instance, use "random" instead of inputting an array.

The seed region generation method specifies which algorithm is used to generate regions in each iteration. This can take the values "ucs", "maxsat", or "incrmaxsat". The details of these approaches are provided in the thesis.

The following command performs the experiment performed in the thesis. All models in the `./models` folder are benchmarked in a multiprocessed manner.

```
python xregions.py -m <any> --benchmark-all
```

Note that the `-m` flag has no effect on the result. All models are benchmarked.


## 3. References

Liffiton, M.H., Previti, A., Malik, A. et al. Fast, flexible MUS enumeration. Constraints 21, 223–250 (2016). https://doi.org/10.1007/s10601-015-9183-0

Ignatiev, A., Izza, Y., Stuckey, P. J., & Marques-Silva, J. (2022). Using MaxSAT for Efficient Explanations of Tree Ensembles. Proceedings of the AAAI Conference on Artificial Intelligence, 36(4), 3776-3785. https://doi.org/10.1609/aaai.v36i4.20292

Ignatiev, A., Narodytska, N., Asher, N., & Marques-Silva, J. (2020). On Relating 'Why?' and 'Why Not?' Explanations. ArXiv, abs/2012.11067.

