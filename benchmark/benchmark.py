import os
import sys
import json
import logging
import random
from multiprocessing import Process, Pool
from multiprocessing.context import TimeoutError

import psutil

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

def benchmark(name, seed_gen, seed=SEED):
    random.seed(seed)
    logging.info(f"Benchmarking model {name} seed_gen {seed_gen}")
    with open(f"models/{name}.json", "r") as fd:
        model = json.load(fd)
    model = Model(model)
    
    lims = get_lims(f"models/{name}.lims")
    logging.info(f"successfully initialised domain limits models/{name}.lims")

    program = ExplanationProgram(model, limits=lims, seed_gen=seed_gen)
    x = random_x(lims)
    with open(f"data/{name}_{seed_gen}.csv", "w") as f:
        f.write("seed_gen_t,lattice_traversal_t,total_t,cum_solver_calls,cum_entailing,cum_nonentailing,seed_entailing,seed_score,max_score,rss_bytes,vms_bytes\n")
        info = psutil.Process().memory_info()
        f.write(f"{program._seed_gen_t},{program._traversal_t},{program._seed_gen_t+program._traversal_t},{program._sat_calls},{program.n_entailing},{program.n_nonentailing},{program.seed_entailing},{program.seed_score},{program.max_score},{info.rss},{info.vms}\n")
        for r in program.enumerate_explanations(x, block_score=False):
            info = psutil.Process().memory_info()
            f.write(f"{program._seed_gen_t},{program._traversal_t},{program._seed_gen_t+program._traversal_t},{program._sat_calls},{program.n_entailing},{program.n_nonentailing},{program.seed_entailing},{program.seed_score},{program.max_score},{info.rss},{info.vms}\n")
            f.flush()
        info = psutil.Process().memory_info()
        f.write(f"{program._seed_gen_t},{program._traversal_t},{program._seed_gen_t+program._traversal_t},{program._sat_calls},{program.n_entailing},{program.n_nonentailing},{program.seed_entailing},{program.seed_score},{program.max_score},{info.rss},{info.vms}\n")
        f.flush()
    logging.info(f"Benchmark complete")

def benchmark_multiprocess(models, seed_gen, seed=SEED):
    with Pool() as pool:
        results = [pool.apply_async(benchmark, (model, seed_gen, seed)) for model in models]
        try:
            [res.wait(timeout=21600) for res in results]
        except TimeoutError:
            pass

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

def benchmark_all(seed_gen, seed=SEED):
    models = sorted(get_models())
    benchmark_multiprocess(models, seed_gen, seed)
