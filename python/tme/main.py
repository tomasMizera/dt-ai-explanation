import os
import argparse

from tme import experiments

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Text Models Explanator")
    argparser.add_argument("--experiment", "-e", help="run experiments, provide name of experiment: c, f, q")
    args = argparser.parse_args()

    if args.experiment:
        if args.experiment == 'c':
            experiments.test_experiment.TestExperiment(os.cpu_count()).run()
        elif args.experiment == 'ray':
            experiments.ray_experiment.run()
