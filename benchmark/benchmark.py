import os
import sys
import json
import time
import logging
import random
from multiprocessing import Process

from src.model import Model
from src.explainer import ExplanationProgram

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(levelname)s|%(asctime)s] %(message)s", 
    datefmt="%m/%d/%Y %I:%M:%S %p"
)
SEED = 21023

def random_x(lims):
    return [random.uniform(l[0], l[1]) for l in lims.values()]

def benchmark_enumerate(name, seed=SEED):
    random.seed(seed)
    logging.info(f"Benchmarking model {name} enumeration...")
    with open(f"models/{name}.json", "r") as fd:
        model = json.load(fd)
    model = Model(model)
    
    lims = get_lims(f"models/{name}.lims")
    logging.info(f"successfully initialised domain limits models/{name}.lims")

    program = ExplanationProgram(model, limits=lims)
    x = random_x(lims)
    with open(f"data/{name}_enumerate.csv", "w") as f:
        f.write("seed_gen_t,lattice_traversal_t,total_t,cum_solver_calls,cum_entailing,cum_nonentailing,max_score\n")
        for r in program.enumerate_explanations(x, block_score=False):
            f.write(f"{program._seed_gen_t},{program._traversal_t},{program._seed_gen_t+program._traversal_t},{program._sat_calls},{program.n_entailing},{program.n_nonentailing},{program.max_score}\n")
            f.flush()
    logging.info(f"Benchmark complete")

def benchmark_explain(name, seed=SEED):
    random.seed(seed)
    logging.info(f"Benchmarking model {name} individual explanations...")
    with open(f"models/{name}.json", "r") as fd:
        model = json.load(fd)
    model = Model(model)
    
    lims = get_lims(f"models/{name}.lims")
    logging.info(f"successfully initialised domain limits models/{name}.lims")

    program = ExplanationProgram(model, limits=lims)
    N = 100
    with open(f"data/{name}_explain.csv", "w") as f:
        f.write("time_taken,solver_calls\n")
        for i in range(N):
            x = random_x(lims)
            program.explain(x)
            f.write(f"{program._explain_t},{program._sat_calls}\n")
            program.reset()
            if i % (N // 10) == 0:
                logging.info(f"Benchmark {100*round(i/N, 2)}% ({i}/{N}) complete...")
    logging.info(f"Benchmark complete")

def benchmark_max(name, seed=SEED):
    random.seed(seed)
    logging.info(f"Benchmarking model {name} maximum volume explanation computation...")
    with open(f"models/{name}.json", "r") as fd:
        model = json.load(fd)
    model = Model(model)
    
    lims = get_lims(f"models/{name}.lims")
    logging.info(f"successfully initialised domain limits models/{name}.lims")

    program = ExplanationProgram(model, limits=lims, seed_gen="max")
    x = random_x(lims)
    with open(f"data/{name}_maxsat.csv", "w") as f:
        f.write("seed_gen_t,lattice_traversal_t,total_t,cum_solver_calls,cum_entailing,cum_nonentailing,seed_score,max_score\n")
        for r in program.enumerate_explanations(x, block_score=False):
            f.write(f"{program._seed_gen_t},{program._traversal_t},{program._seed_gen_t+program._traversal_t},{program._sat_calls},{program.n_entailing},{program.n_nonentailing},{program.seed_score},{program.max_score}\n")
            f.flush()
        f.write(f"{program._seed_gen_t},{program._traversal_t},{program._seed_gen_t+program._traversal_t},{program._sat_calls},{program.n_entailing},{program.n_nonentailing},{program.seed_score},{program.max_score}\n")
        f.flush()
    logging.info(f"Benchmark complete")

def benchmark_greedy(name, seed=SEED):
    random.seed(seed)
    logging.info(f"Benchmarking model {name} maximum volume explanation computation...")
    with open(f"models/{name}.json", "r") as fd:
        model = json.load(fd)
    model = Model(model)
    
    lims = get_lims(f"models/{name}.lims")
    logging.info(f"successfully initialised domain limits models/{name}.lims")

    program = ExplanationProgram(model, limits=lims, seed_gen="greedy")
    x = random_x(lims)
    with open(f"data/{name}_greedy.csv", "w") as f:
        f.write("seed_gen_t,lattice_traversal_t,total_t,cum_solver_calls,cum_entailing,cum_nonentailing,seed_score,max_score\n")
        for r in program.enumerate_explanations(x, block_score=False):
            f.write(f"{program._seed_gen_t},{program._traversal_t},{program._seed_gen_t+program._traversal_t},{program._sat_calls},{program.n_entailing},{program.n_nonentailing},{program.seed_score},{program.max_score}\n")
            f.flush()
        f.write(f"{program._seed_gen_t},{program._traversal_t},{program._seed_gen_t+program._traversal_t},{program._sat_calls},{program.n_entailing},{program.n_nonentailing},{program.seed_score},{program.max_score}\n")
        f.flush()
    logging.info(f"Benchmark complete")

def get_models(model_dir="models"):
    return set(map(lambda x: x.split(".")[0], os.listdir(model_dir)))

def get_lims(fname):
    lims = {}
    with open(fname, "r") as f:
        line = f.readline()
        while line:
            line = line.split(",")
            lims[int(line[0])] = (float(line[1]), float(line[2]))
            line = f.readline()
    return lims

def benchmark_all():
    models = get_models()
    for model in models:
        with open(f"models/{model}.json", "r") as fd:
            model_obj = Model(json.load(fd))
        p_max = Process(target=benchmark_max, name=f"benchmark_{model}_max", args=(model,))
        p_greedy = Process(target=benchmark_greedy, name=f"benchmark_{model}_greedy", args=(model,))
        ps = [p_max, p_greedy]
        [p.start() for p in ps]

        timeout = 1200
        timeout_t = time.strftime("%H:%M:%S", time.localtime(time.time() + timeout))
        s = 0
        logging.info(f"Benchmarking {model} - timeout in 1200s ({timeout_t})")
        logging.info(
            "\nPROGRAM INFO:\n" + \
                f"\tClasses: {model_obj.num_output_group}\n" + \
                f"\tFeatures: {model_obj.num_feature}\n" + \
                f"\tTrees: {model_obj.num_trees}"
        )
        while s < timeout and any([p.is_alive() for p in ps]):
            s += 1
            time.sleep(1)
        if any([p.is_alive() for p in ps]):
            logging.info(f"Timeout - killing all processes...")
            [p.kill() for p in ps]
            [p.join() for p in ps]
            logging.info(f"Killed children")
        logging.info(f"Releasing resources...")
        [p.close() for p in ps]
        logging.info(f"Benchmarking for {model} complete.")

