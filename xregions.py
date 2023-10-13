import json
import sys
import time
import argparse
import logging
import random

from src.model import Model
from src.explainer import ExplanationProgram
from benchmark.benchmark import benchmark_all

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(levelname)s|%(asctime)s] %(message)s", 
    datefmt="%m/%d/%Y %I:%M:%S %p"
)

SEED = 21023

def get_lims(fname):
    lims = {}
    with open(fname, "r") as f:
        line = f.readline()
        while line:
            line = line.split(",")
            lims[int(line[0])] = (float(line[1]), float(line[2]))
            line = f.readline()
    return lims

def random_x(lims):
    return [random.uniform(l[0], l[1]) for l in lims.values()]

def main():
    parser = argparse.ArgumentParser(
        description="Demonstration for loading and printing XGBoost model.")
    parser.add_argument("-m", "--model",
                        type=str,
                        required=True,
                        help="Name of a model in the models folder.")
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("-E", "--enumerate",
                        type=str,
                        help="Enumerate all explanations for an instance.")
    action_group.add_argument("-e", "--explain",
                        type=str,
                        help="Generate one explanation for a given instance.")
    action_group.add_argument("--benchmark-explain",
                        action="store_true",
                        required=False,
                        help="Run benchmark for explaining.")
    action_group.add_argument("--benchmark-enumerate",
                        action="store_true",
                        required=False,
                        help="Run benchmark for enumerating.")
    action_group.add_argument("--benchmark-maxvol",
                        action="store_true",
                        required=False,
                        help="Run benchmark for finding maximum volume region.")
    parser.add_argument("--loglevel",
                        type=str,
                        required=False,
                        help="Program logging level.")
    parser.add_argument("--block-score",
                        type=bool,
                        default=False,
                        required=False,
                        help="Whether or not to block score when enumerating.")
    parser.add_argument("--seed-gen",
                        type=str,
                        default="rand",
                        required=False,
                        help="Seed generation method: (rand|min|max)")
    args = parser.parse_args()

    if args.benchmark_maxvol:
        benchmark_all()
        return
    if args.benchmark_explain:
        benchmark_explain(args.model)
        return
    if args.benchmark_enumerate:
        benchmark_enumerate(args.model)
        return

    if args.loglevel:
        numeric_level = getattr(logging, args.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError("Invalid log level: %s" % args.loglevel)
        logging.getLogger().setLevel(numeric_level)

    with open(f"models/{args.model}.json", "r") as fd:
        model = json.load(fd)
    model = Model(model)
    logging.info(f"successfully initialised models/{args.model}.json")
    
    lims = get_lims(f"models/{args.model}.lims")
    logging.info(f"successfully initialised domain limits models/{args.model}.json")
    
    seed_gen = args.seed_gen
    block_score = args.block_score

    instance = args.explain if args.explain is not None else args.enumerate
    if instance == "random":
        random.seed(SEED)
        instance = random_x(lims)
    else:
        instance = [float(x) for x in instance.split(",")]

    program = ExplanationProgram(model, limits=lims, seed_gen=seed_gen)
    logging.info(
        "\nPROGRAM INFO:\n" + \
            f"\tObjective: {model.objective}\n"
            f"\tClasses: {2 if 'binary' in model.objective else model.num_output_group}\n" + \
            f"\tFeatures: {model.num_feature}\n" + \
            f"\tTrees: {model.num_trees}\n" + \
            f"\tSeed Generation: {seed_gen}\n" + \
            f"\tThresholds: {program.fs_info.n_thresholds()}\n" + \
            f"\tPairs: {program.fs_info.n_pairs()}\n" + \
            f"\tPossible Regions: {program.fs_info.n_regions()}"
    )

    c = program.entailer.predict(instance)
    if args.explain is not None:
        logging.info(f"EXPLAIN: {instance} -> {c} | block_score: {block_score}")
        r = program.explain(instance)
        logging.info(f"COMPLETE:\n{r}")
    elif args.enumerate is not None:
        logging.info(
            "THRESHOLDS:\n" + \
            "\n".join([f"{i}: {program.fs_info.get_domain(i)}" for i in sorted(model.thresholds.keys())])
        )
        logging.info(f"ENUMERATE EXPLANATIONS: {instance} -> {c} | block_score: {block_score}")
        for r in program.enumerate_explanations(instance, block_score=block_score):
            pass


if __name__ == "__main__":
    main()
