import argparse
import logging
import os
import pandas as pd
import numpy as np

from graph import Graph
from parser import obo_parser, gt_parser, pred_parser, ia_parser
from evaluation import get_leafs_idx, get_roots_idx, evaluate_prediction


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CAFA-evaluator. Calculate precision-recall curves and F-max / S-min')

    parser.add_argument('obo_file', help='Ontology file, OBO format')
    parser.add_argument('pred_dir', help='Predictions directory. Sub-folders are iterated recursively')
    parser.add_argument('gt_file', help='Ground truth file')

    parser.add_argument('-out_dir', default='results',
                        help='Output directory. By default it creates \"results/\" in the current directory')
    parser.add_argument('-ia', help='Information accretion file (columns: <term> <information_accretion>)')
    parser.add_argument('-no_orphans', action='store_true', default=False,
                        help='Consider terms without parents, e.g. the root(s), in the evaluation')
    parser.add_argument('-norm', choices=['cafa', 'pred', 'gt'], default='cafa',
                        help='Normalization strategy. i) CAFA strategy (cafa); '
                             'ii) consider predicted targets (pred); '
                             'iii) consider ground truth proteins (gt)')
    parser.add_argument('-prop', choices=['max', 'fill'], default='max',
                        help='Ancestor propagation strategy. i) Propagate the max score of the traversed subgraph '
                             'iteratively (max); ii) Propagate with max until a different score is found (fill)')
    parser.add_argument('-th_step', type=float, default=0.01,
                        help='Threshold step size in the range [0, 1]. A smaller step, means more calculation.')
    parser.add_argument('-max_terms', type=int, default=None,
                        help='Number of terms for protein and namespace to consider in the evaluation.')
    parser.add_argument('-threads', type=int, default=4,
                        help='Parallel threads. 0 means use all available CPU threads. '
                             'Do not use multithread if you are short in memory')

    args = parser.parse_args()

    # Create output folder here in order to store the log file
    out_folder = os.path.normpath(args.out_dir) + "/"
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    # Set the logger
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    rootLogger = logging.getLogger()
    # rootLogger.setLevel(logging.DEBUG)
    rootLogger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler("{0}/info.log".format(out_folder))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    # Parse and set information accretion (optional)
    ia_dict = None
    if args.ia is not None:
        ia_dict = ia_parser(args.ia)

    # Parse the OBO file and creates a different graph for each namespace
    ontologies = []
    for ns, terms_dict in obo_parser(args.obo_file).items():
        ontologies.append(Graph(ns, terms_dict, ia_dict, not args.no_orphans))
        logging.info("Ontology: {}, roots {}, leaves {}".format(ns, len(get_roots_idx(ontologies[-1].dag)), len(get_leafs_idx(ontologies[-1].dag))))

    # Set prediction files
    pred_folder = os.path.normpath(args.pred_dir) + "/"  # add the tailing "/"
    pred_files = []
    for root, dirs, files in os.walk(pred_folder):
        for file in files:
            pred_files.append(os.path.join(root, file))
    logging.debug("Prediction paths {}".format(pred_files))

    # Parse ground truth file
    gt = gt_parser(args.gt_file, ontologies)

    # Tau array, used to compute metrics at different score thresholds
    tau_arr = np.arange(0.01, 1, args.th_step)

    # Parse prediction files and perform evaluation
    dfs = []
    for file_name in pred_files:
        print("Started the prediction")
        prediction = pred_parser(file_name, ontologies, gt, args.prop, args.max_terms)
        df_pred = evaluate_prediction(prediction, gt, ontologies, tau_arr, args.norm, args.threads)
    print('Done with prediction')
    df_pred.to_csv('evaluation.csv')
       
