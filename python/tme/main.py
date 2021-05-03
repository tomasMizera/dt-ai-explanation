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
                conf = load(f, Loader=Loader)
        if args.experiment == 'mptest':
            experiments.test_experiment.TestExperiment(os.cpu_count()).run()
        elif args.experiment == 'raytest':
            experiments.ray_experiment.run()
        elif args.experiment == 'quantitative' or args.experiment == 'q':
            if not conf:
                raise FileNotFoundError('Missing config file or config file could not be read. Required for q.')

            if conf['mode'] == 'precomputing':
                # Here we are precomputing
                experiments.quantitative.QuantitativeExperiment(args.experimenttag).run(precompute=True, **conf)
            else:
                # Here we are building summaries
                experiments.quantitative.QuantitativeExperiment(args.experimenttag).run(precompute=False, **conf)
        elif args.experiment == 'limebench':
            experiments.lime_benchmark.run()
        elif args.experiment == 'user-study' or args.experiment == 'u':
            experiments.user_study.run(
                conf['datapath'],
                conf['modelpath'],
                conf['vectorizerpath'],
                conf['outpath'],
                conf['workers']
            )
        else:
            print(f'Uknown test name {args.experiment}')
