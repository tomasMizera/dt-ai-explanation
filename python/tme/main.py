import os
import argparse

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
        "--setup",
        "-s",
        help="arguments to quantitative experiment in format batchsize-minsentencescount-fp",
        required=False
    )
    args = argparser.parse_args()

    if args.experiment:
        if args.experiment == 'mptest':
            experiments.test_experiment.TestExperiment(os.cpu_count()).run()
        elif args.experiment == 'raytest':
            experiments.ray_experiment.run()
        elif args.experiment == 'quantitative' or args.experiment == 'q':
            eargs = args.setup.split('-') if args.setup else [10, 40, 10, 5, 'f', 't']
            isTuning = eargs[4] == 't'
            precomputeOnly = eargs[5] == 't'
            experiments.quantitative.QuantitativeExperiment(args.experimenttag).run(
                instances=int(eargs[0]),
                from_sentences_count=int(eargs[1]),
                factor=int(eargs[2]),
                run_to_factor=isTuning,
                only_precompute=precomputeOnly,
                batch_size=int(eargs[3])
            )
        elif args.experiment == 'limebench':
            experiments.lime_benchmark.run()
        else:
            print(f'Uknown test name {args.experiment}')
