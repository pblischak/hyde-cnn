#!/usr/bin/env python3

"""
<< run_sims.py >>

Simulate data sets from admixture/hybridization models.

Arguments
---------

For more specific details, type: run_sims.py -h

    - model <string> : name of model to run ['no_hybridization','hybrid_speciation','admixture']
    - cu    <float>  : branch scaling in coalescent units

Output
------

Writes a compressed numpy array with simulated values of nucleotide diversity (pi)
in 1000 windows across all pairwise comparisons of taxa and individuals.
"""

from models import (
    no_hybridization,
    hybrid_speciation,
    admixture,
    admixture_w_gflow
)
from tensorflow.keras.models import load_model
from summary_stats import abba_baba_stats
import numpy as np
import tskit as tsk
from sys import argv, exit
import argparse

def open_file(filename, mode):
    try:
        handle = open(filename, mode)
    except:
        print("Error: Could not open {}...".format(filename))
        exit(-1)
    return handle

def print_params(params, f_out):
    for i in range(len(params)-1):
        print("{},".format(params[i]), end='', file=f_out)
    print(params[-1], file=f_out)
    if (sim+1) % 50 == 0: f_out.flush()

if __name__ == "__main__":
    """
    Run the script from the command line.
    """
    # Print docstring if only the name of the script is given
    if len(argv) < 2:
        print(__doc__)
        exit(0)

    #
    sim_models = [
        "no_hyb",
        "hyb_sp",
        "admix",
        "admix_mig"
    ]

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Options for run_sims.py", add_help=True)

    required = parser.add_argument_group("required arguments")
    required.add_argument('-m', '--model', action="store", type=str, required=True,
                          metavar='\b',
                          help="name of model to simulate ['no_hybridization','hybrid_speciation','admixture','admixture_w_gflow']")
    additional = parser.add_argument_group("additional arguments")
    additional.add_argument('-cu','--coal_units', action="store", type=float, default=1.0,
                            metavar='\b', help="branch scaling in coalescent units")

    args       = parser.parse_args()
    model_name = args.model
    cu         = args.coal_units

    if model_name == "no_hybridization":
        print("Running simulations for no_hybridization model...", flush=True)
        model = no_hybridization
        model_index = 0
        model_out  = open_file("{}_{}_tests.csv".format(model_name, cu), 'w')
        print("time_units,L,tau1,tau2,tau3,mu,r,true_model_weight,best_model,best_model_weight,D,f_hom,D_p", file=model_out, flush=True)
    elif model_name == "hybrid_speciation":
        print("Running simulations for hybrid_speciation model...", flush=True)
        model = hybrid_speciation
        model_index = 1
        model_out  = open_file("{}_{}_tests.csv".format(model_name, cu), 'w')
        print("time_units,L,tau1,tau2,tau3,mu,r,gamma,true_model_weight,best_model,best_model_weight,D,f_hom,D_p", file=model_out, flush=True)
    elif model_name == "admixture":
        print("Running simulations for admixture model...", flush=True)
        model = admixture
        model_index = 2
        model_out  = open_file("{}_{}_tests.csv".format(model_name, cu), 'w')
        print("time_units,L,tau1,tau2,tau3,mu,r,gamma,admix_time,true_model_weight,best_model,best_model_weight,D,f_hom,D_p", file=model_out, flush=True)
    elif model_name == "admixture_w_gflow":
        print("Running simulations for admixture_w_gflow model...", flush=True)
        model = admixture_w_gflow
        model_index = 3
        model_out  = open_file("{}_{}_tests.csv".format(model_name, cu), 'w')
        print("time_units,L,tau1,tau2,tau3,mu,r,gamma,admix_time,m,true_model_weight,best_model,best_model_weight,D,f_hom,D_p", file=model_out, flush=True)
    else:
        print("Invalid model name supplied: {}".format(model_name), flush=True)
        print("The following models are currently implemented:", flush=True)
        print("  no_hybridization", flush=True)
        print("  hybrid_speciation", flush=True)
        print("  admixture", flush=True)
        print("  admixture_w_gflow", flush=True)
        exit(-1)

    # Specify the number of simulations
    nsims = 10000

    # Define sample sets
    samples_P1_P2  = [[i, j+5] for i in range(5) for j in range(5)]
    samples_P1_P3  = [[i, j+10] for i in range(5) for j in range(5)]
    samples_P2_P3  = [[i+5, j+10] for i in range(5) for j in range(5)]
    samples_P1_Out = [[i, j+15] for i in range(5) for j in range(5)]
    samples_P2_Out = [[i+5, j+15] for i in range(5) for j in range(5)]
    samples_P3_Out = [[i+10, j+15] for i in range(5) for j in range(5)]

    # Specify the number of windows to calculate pi
    nwindows = 1000

    # Specify the print frequency for tracking completed simulations
    # Set to None if you don't want to print
    print_freq = 1000

    # Define the numpy array that will hold the simulated
    # windows of pi. The size of the last dimensions (6) comes
    # from the fact that we are working with 4 species 2 at a time:
    # 4 choose 2 = 6.
    model_data = np.zeros((1,nwindows, 6, 1))
    trained_cnn = load_model('hyde_cnn_min_{}.mdl'.format(cu))

    for sim in range(nsims):
        """
        This is the main loop for doing the simulaitons.
        """
        model_params, model_ts = model(cu)
        windows = np.linspace(0,model_params[1],nwindows+1)
        d_stats = abba_baba_stats(model_ts.genotype_matrix().T)

        model_data[0,:,0,0] = np.min(model_ts.diversity(samples_P1_P2, windows=windows).T, axis=0)
        model_data[0,:,1,0] = np.min(model_ts.diversity(samples_P1_P3, windows=windows).T, axis=0)
        model_data[0,:,2,0] = np.min(model_ts.diversity(samples_P2_P3, windows=windows).T, axis=0)
        model_data[0,:,3,0] = np.min(model_ts.diversity(samples_P1_Out, windows=windows).T, axis=0)
        model_data[0,:,4,0] = np.min(model_ts.diversity(samples_P2_Out, windows=windows).T, axis=0)
        model_data[0,:,5,0] = np.min(model_ts.diversity(samples_P3_Out, windows=windows).T, axis=0)

        pred = trained_cnn.predict(model_data / np.max(model_data))

        print_params(
            model_params +
            [pred[0][model_index], sim_models[pred[0].argmax()], pred[0][pred[0].argmax()]] +
            [v for _,v in d_stats.items()],
            model_out
        )

        if print_freq is not None and (sim+1) % print_freq == 0:
            print("Completed {} simulation iterations...".format(sim+1), flush=True)

    print("Done.", flush=True)
