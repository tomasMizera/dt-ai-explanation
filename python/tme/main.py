import os
import argparse
from yaml import load, Loader

from tme import experiments

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Text Models Explanator")
    argparser.add_argument(
        "--experiment",
        "-e",
        help="run experiments, provide name of experiment: mptest, raytest, quantitative(q), limebench",
        required=False
    )
    argparser.add_argument(
        "--experimenttag",
        "-t",
        help="name of the experiment (visible in logs and outputs)",
        required=False
    )
    argparser.add_argument(
        "--conffile",
        "-c",
        help="path to config yaml file",
        required=False
    )
    args = argparser.parse_args()

    if args.experiment:
        if args.conffile:
            with open(args.conffile, 'r') as f:
                config = load(f, Loader=Loader)
        if args.experiment == 'mptest':
            experiments.test_experiment.TestExperiment(os.cpu_count()).run()
        elif args.experiment == 'raytest':
            experiments.ray_experiment.run()
        elif args.experiment == 'quantitative' or args.experiment == 'q':
            if not config:
                raise FileNotFoundError('Missing config file or config file could not be read. Required for q.')

            if config['precomputing']['precompute']:
                # Here we are precomputing
                experiments.quantitative.QuantitativeExperiment(args.experimenttag).run(
                    precompute=True,
                    instances=config['precomputing']['instances'],
                    worker_load=config['precomputing']['worker_load'],
                    from_sentences_count=config['precomputing']['min_sentences_count'],
                    workers=config['workers']
                )
            else:
                # Here we are building summaries
                experiments.quantitative.QuantitativeExperiment(args.experimenttag).run(
                    precompute=False,
                    path_to_precomputed=config['summarizing']['precomputed_dir'],
                    factor=config['summarizing']['factor'],
                    run_to_factor=config['summarizing']['run_to_factor'],
                    workers=config['workers']
                )
        elif args.experiment == 'limebench':
            experiments.lime_benchmark.run()
        else:
            print(f'Uknown test name {args.experiment}')
