#!/usr/bin/env python3

"""
<< run_sims.py >>

Simulate data sets from admixture/hybridization models.

Arguments
---------

For more specific details, type: run_sims.py -h

    - model <string> : name of model to run ['no_hybridization','admixture','gene_flow']
    - rep    <float> : replicate number for simulation

Output
------

Writes a compressed numpy array with simulated values of nucleotide diversity (pi)
in 950 windows across all pairwise comparisons of taxa and individuals.
"""

from heliconius_models import (
    no_hybridization,
    admixture,
    gene_flow
)
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
    print(",".join([str(p) for p in params]), file=f_out, flush=True)

def get_recomb_rates(mapFile):
    """
    Simple function to get recombination rates from crossover rates TSV file.
    """
    with open(mapFile) as f:
        lines = [line.strip() for line in f.readlines()]
        pos   = [int(l.split()[1])-200001 for l in lines[2:-2]]+[9500000]
        rates = [float(l.split()[-1])/1e08 for l in lines[2:-2]]+[0]
    return (pos,rates)

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
                          help="name of model to simulate ['no_hybridization','hybrid_speciation','admixture','gene_flow']")
    required.add_argument('-r', '--rep', action="store", type=str, required=True,
                          metavar='\b',
                          help="replicate number for simulaiton")

    args       = parser.parse_args()
    model_name = args.model
    rep        = args.rep

    if model_name == "no_hybridization":
        print("Running simulations for no_hybridization model...", flush=True)
        model = no_hybridization
    elif model_name == "admixture":
        print("Running simulations for admixture model...", flush=True)
        model = admixture
    elif model_name == "gene_flow":
        print("Running simulations for gene_flow model...", flush=True)
        model = gene_flow
    else:
        print("Invalid model name supplied: {}".format(model_name), flush=True)
        print("The following models are currently implemented:", flush=True)
        print("  no_hybridization", flush=True)
        print("  admixture", flush=True)
        print("  gene_flow", flush=True)
        exit(-1)

    # Specify the number of simulations
    nsims = 5000

    # Define sample sets
    samples_P1_P2  = [[i, j+10] for i in range(10) for j in range(10)]
    samples_P1_P3  = [[i, j+20] for i in range(10) for j in range(10)]
    samples_P2_P3  = [[i+10, j+20] for i in range(10) for j in range(10)]
    samples_P1_Out = [[i, j+30] for i in range(10) for j in range(4)]
    samples_P2_Out = [[i+10, j+30] for i in range(10) for j in range(4)]
    samples_P3_Out = [[i+20, j+30] for i in range(10) for j in range(4)]

    # Specify the number of windows to calculate pi
    nwindows = 950
    windows  = np.linspace(0,9500000,nwindows+1)

    # Specify the print frequency for tracking completed simulations
    # Set to None if you don't want to print
    print_freq = 500

    # Open CSV files for recording parameters used to simulate data
    model_out  = open_file("heliconius_{}_{}.csv".format(model_name,rep), 'w')

    # Define the numpy array that will hold the simulated
    # windows of pi. The size of the last dimensions (6) comes
    # from the fact that we are working with 4 species 2 at a time:
    # 4 choose 2 = 6.
    model_mean = np.zeros((nsims, 950, 6))
    model_min  = np.zeros((nsims, 950, 6))

    # Get recombination map
    pos,rates = get_recomb_rates("chr5_recombRates.tsv")

    for sim in range(nsims):
        """
        This is the main loop for doing the simulaitons.
        """
        model_params, model_ts = model(pos,rates)

        model_mean[sim,:,0] = np.mean(model_ts.diversity(samples_P1_P2, windows=windows).T, axis=0)
        model_min[sim,:,0]  = np.min(model_ts.diversity(samples_P1_P2, windows=windows).T, axis=0)

        model_mean[sim,:,1] = np.mean(model_ts.diversity(samples_P1_P3, windows=windows).T, axis=0)
        model_min[sim,:,1]  = np.min(model_ts.diversity(samples_P1_P3, windows=windows).T, axis=0)

        model_mean[sim,:,2] = np.mean(model_ts.diversity(samples_P2_P3, windows=windows).T, axis=0)
        model_min[sim,:,2]  = np.min(model_ts.diversity(samples_P2_P3, windows=windows).T, axis=0)

        model_mean[sim,:,3] = np.mean(model_ts.diversity(samples_P1_Out, windows=windows).T, axis=0)
        model_min[sim,:,3]  = np.min(model_ts.diversity(samples_P1_Out, windows=windows).T, axis=0)

        model_mean[sim,:,4] = np.mean(model_ts.diversity(samples_P2_Out, windows=windows).T, axis=0)
        model_min[sim,:,4]  = np.min(model_ts.diversity(samples_P2_Out, windows=windows).T, axis=0)

        model_mean[sim,:,5] = np.mean(model_ts.diversity(samples_P3_Out, windows=windows).T, axis=0)
        model_min[sim,:,5]  = np.min(model_ts.diversity(samples_P3_Out, windows=windows).T, axis=0)

        snp_matrix = model_ts.genotype_matrix().T
        stat_dict  = abba_baba_stats(snp_matrix)
        print_params([snp_matrix.shape[1]]+list(stat_dict.values()), model_out)
        if print_freq is not None and (sim+1) % print_freq == 0:
            print("Completed {} simulation iterations...".format(sim+1), flush=True)

    np.savez_compressed('heliconius_{}_{}.npz'.format(model_name, rep), mean=model_mean, min=model_min)
    print("Done.", flush=True)
