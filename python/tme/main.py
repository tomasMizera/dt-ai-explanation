import os
import argparse

from tme import experiments

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Text Models Explanator")
    argparser.add_argument("--experiment", "-e", help="run experiments, provide name of experiment: c, f, q")
    argparser.add_argument(
        "--experimenttag",
        "-t",
        help="name of the experiment (visible in logs and outputs)",
        required=False
    )
    args = argparser.parse_args()

    if args.experiment:
        if args.experiment == 'c':
            experiments.test_experiment.TestExperiment(os.cpu_count()).run()
        elif args.experiment == 'ray':
            experiments.ray_experiment.run(redis_password='5241590000000000')
        elif args.experiment == 'fp':
            experiments.find_factor.FindFactorExperiment(args.experimenttag).run(
                batch_size=10,
                to_factor=7,
                from_sentences_count=40
            )
        elif args.experiment == 'limebench':
            experiments.lime_benchmark.run()

