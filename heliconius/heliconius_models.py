#!/usr/bin/env python3

import msprime as msp
import numpy as np
from math import floor

def no_hybridization(positions,rates,mu=7.7e-07):
    """
    num    cyd  mel-W mel-E
     |      |     |     |
     |      |     |     |
     |      |     |-----|
     |      |           |
     |      |           |
     |      |-----------|
     |                  |
     |                  |
     |                  |
     |                  |
     |------------------|
               |


    """
    # Coalescent units are the number of 2N generations
    time_units = 2.0 * 2000.0

    # Divergence times
    tau1 = 0.5
    tau2 = 1.5
    tau3 = 4.0

    # Recombination rates
    rmap = msp.RecombinationMap(positions,rates)

    # Set up population configurations
    population_configurations = [
        msp.PopulationConfiguration(
            sample_size=10,
            initial_size=2000
        ),
        msp.PopulationConfiguration(
            sample_size=10,
            initial_size=2000
        ),
        msp.PopulationConfiguration(
            sample_size=10,
            initial_size=2000
        ),
        msp.PopulationConfiguration(
            sample_size=4,
            initial_size=2000
        )
    ]

    # Set up demographic events
    demographic_events = [
        msp.MassMigration(
            time = tau1 * time_units,
            source = 0, destination=1,
            proportion=1.0
        ),
        msp.MassMigration(
            time = tau2 * time_units,
            source = 1, destination=2,
            proportion=1.0
        ),
        msp.MassMigration(
            time = tau3 * time_units,
            source = 2, destination=3,
            proportion=1.0
        )
    ]

    return ([time_units, tau1, tau2, tau3, mu],
            msp.simulate(mutation_rate=mu, recombination_map=rmap,
                         population_configurations=population_configurations,
                         demographic_events=demographic_events))

def admixture(positions,rates,mu=7.7e-07):
    """
    num    cyd  mel-W mel-E
     |      |     |     |
     |      |---->|     |
     |      |     |-----|
     |      |           |
     |      |           |
     |      |-----------|
     |                  |
     |                  |
     |                  |
     |                  |
     |------------------|
               |


    """
    # Coalescent units are the number of 2N generations
    time_units = 2.0 * 2000.0

    # Divergence times
    tau1 = 0.5
    tau2 = 1.5
    tau3 = 4.0

    # Recombination rates
    rmap = msp.RecombinationMap(positions,rates)

    # Draw hybridization fraction
    gamma = np.random.uniform(0.3, 0.4)

    # Draw admixture time, restrict to occurring between
    # and 50% and ~100% of tau1
    admix_time = np.random.uniform(0.5 * tau1, tau1+1e-10)

    # Set up population configurations
    population_configurations = [
        msp.PopulationConfiguration(
            sample_size=10,
            initial_size=2000
        ),
        msp.PopulationConfiguration(
            sample_size=10,
            initial_size=2000
        ),
        msp.PopulationConfiguration(
            sample_size=10,
            initial_size=2000
        ),
        msp.PopulationConfiguration(
            sample_size=4,
            initial_size=2000
        )
    ]

    # Set up demographic events
    demographic_events = [
        msp.MassMigration(
            time = admix_time * time_units,
            source = 1, destination = 2,
            proportion = gamma
        ),
        msp.MassMigration(
            time = tau1 * time_units,
            source = 1, destination = 0,
            proportion = 1.0
        ),
        msp.MassMigration(
            time = tau2 * time_units,
            source = 0, destination = 2,
            proportion = 1.0
        ),
        msp.MassMigration(
            time = tau3 * time_units,
            source = 2, destination = 3,
            proportion = 1.0
        )
    ]

    return ([time_units, tau1, tau2, tau3, mu, gamma, admix_time],
            msp.simulate(mutation_rate=mu, recombination_map=rmap,
                         population_configurations=population_configurations,
                         demographic_events=demographic_events))

def gene_flow(positions,rates,mu=7.7e-07):
    """
    num    cyd  mel-W mel-E
     |      |---->|     |
     |      |---->|     |
     |      |---->|-----|
     |      |           |
     |      |           |
     |      |-----------|
     |                  |
     |                  |
     |                  |
     |                  |
     |------------------|
               |


    """
    # Coalescent units are the number of 2N generations
    time_units = 2.0 * 2000.0

    # Divergence times
    tau1 = 0.5
    tau2 = 1.5
    tau3 = 4.0

    # Recombination rates
    rmap  = msp.RecombinationMap(positions,rates)

    # Migration rate and migration matrix
    f = np.random.uniform(0.3, 0.4)
    m =  f / (tau1 * time_units)
    #print(m)
    #m = 1.0 - np.exp(np.log(1.0 - f) / (tau1 * time_units))
    #print(m)

    migration_matrix = [
        [0, 0, 0, 0],
        [0, 0, m, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    # Set up population configurations
    population_configurations = [
        msp.PopulationConfiguration(
            sample_size=10,
            initial_size=2000
        ),
        msp.PopulationConfiguration(
            sample_size=10,
            initial_size=2000
        ),
        msp.PopulationConfiguration(
            sample_size=10,
            initial_size=2000
        ),
        msp.PopulationConfiguration(
            sample_size=4,
            initial_size=2000
        )
    ]

    # Set up demographic events
    demographic_events = [
        msp.MassMigration(
            time = tau1 * time_units,
            source = 1, destination = 0,
            proportion = 1.0
        ),
        msp.MigrationRateChange(
            time=tau1 * time_units,
            rate=0
        ),
        msp.MassMigration(
            time = tau2 * time_units,
            source = 0, destination = 2,
            proportion = 1.0
        ),
        msp.MassMigration(
            time = tau3 * time_units,
            source = 2, destination = 3,
            proportion = 1.0
        )
    ]

    return ([time_units, tau1, tau2, tau3, mu, m],
            msp.simulate(mutation_rate=mu, recombination_map=rmap,
                         population_configurations=population_configurations,
                         migration_matrix=migration_matrix,
                         demographic_events=demographic_events))
