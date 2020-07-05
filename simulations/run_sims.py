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
import numpy as np
import tskit as tsk
from random import shuffle
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
    elif model_name == "hybrid_speciation":
        print("Running simulations for hybrid_speciation model...", flush=True)
        model = hybrid_speciation
    elif model_name == "admixture":
        print("Running simulations for admixture model...", flush=True)
        model = admixture
    elif model_name == "admixture_w_gflow":
        print("Running simulations for admixture_w_gflow model...", flush=True)
        model = admixture_w_gflow
    else:
        print("Invalid model name supplied: {}".format(model_name), flush=True)
        print("The following models are currently implemented:", flush=True)
        print("  no_hybridization", flush=True)
        print("  hybrid_speciation", flush=True)
        print("  admixture", flush=True)
        print("  admixture_w_gflow", flush=True)
        exit(-1)

    # Specify the number of simulations
    nsims = 20000

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

    # Open CSV files for recording parameters used to simulate data
    model_out  = open_file("{}_{}.csv".format(model_name, cu), 'w')

    # Define the numpy array that will hold the simulated
    # windows of pi. The size of the last dimensions (6) comes
    # from the fact that we are working with 4 species 2 at a time:
    # 4 choose 2 = 6.
    model_mean = np.zeros((nsims, nwindows, 6))
    model_min  = np.zeros((nsims, nwindows, 6))
    #model_std  = np.zeros((nsims, nwindows, 6))

    for sim in range(nsims):
        """
        This is the main loop for doing the simulaitons.
        """
        model_params, model_ts = model(cu)
        print_params(model_params + [round(model_ts.segregating_sites()*model_params[1])], model_out)
        windows = np.linspace(0,model_params[1],nwindows+1)

        model_mean[sim,:,0] = np.mean(model_ts.diversity(samples_P1_P2, windows=windows).T, axis=0)
        model_min[sim,:,0]  = np.min(model_ts.diversity(samples_P1_P2, windows=windows).T, axis=0)
        #model_std[sim,:,0]  = np.std(model_ts.diversity(samples_P1_P2, windows=windows).T, axis=0)

        model_mean[sim,:,1] = np.mean(model_ts.diversity(samples_P1_P3, windows=windows).T, axis=0)
        model_min[sim,:,1]  = np.min(model_ts.diversity(samples_P1_P3, windows=windows).T, axis=0)
        #model_std[sim,:,1]  = np.std(model_ts.diversity(samples_P1_P3, windows=windows).T, axis=0)

        model_mean[sim,:,2] = np.mean(model_ts.diversity(samples_P2_P3, windows=windows).T, axis=0)
        model_min[sim,:,2]  = np.min(model_ts.diversity(samples_P2_P3, windows=windows).T, axis=0)
        #model_std[sim,:,2]  = np.std(model_ts.diversity(samples_P2_P3, windows=windows).T, axis=0)

        model_mean[sim,:,3] = np.mean(model_ts.diversity(samples_P1_Out, windows=windows).T, axis=0)
        model_min[sim,:,3]  = np.min(model_ts.diversity(samples_P1_Out, windows=windows).T, axis=0)
        #model_std[sim,:,3]  = np.std(model_ts.diversity(samples_P1_Out, windows=windows).T, axis=0)

        model_mean[sim,:,4] = np.mean(model_ts.diversity(samples_P2_Out, windows=windows).T, axis=0)
        model_min[sim,:,4]  = np.min(model_ts.diversity(samples_P2_Out, windows=windows).T, axis=0)
        #model_std[sim,:,4]  = np.std(model_ts.diversity(samples_P2_Out, windows=windows).T, axis=0)

        model_mean[sim,:,5] = np.mean(model_ts.diversity(samples_P3_Out, windows=windows).T, axis=0)
        model_min[sim,:,5]  = np.min(model_ts.diversity(samples_P3_Out, windows=windows).T, axis=0)
        #model_std[sim,:,5]  = np.std(model_ts.diversity(samples_P3_Out, windows=windows).T, axis=0)

        if print_freq is not None and (sim+1) % print_freq == 0:
            print("Completed {} simulation iterations...".format(sim+1), flush=True)

    np.savez_compressed('{}_{}.npz'.format(model_name, cu), mean=model_mean, min=model_min)
    print("Done.", flush=True)
