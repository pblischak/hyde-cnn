#!/usr/bin/env python3

import msprime as msp
import numpy as np

def no_hybridization(coal_units=1.0):
    """
    Out    P3    P2    P1
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
    time_units = 2.0 * 1000.0 * coal_units

    # Draw sequence length
    L = np.random.choice([x for x in range(int(1e7), int(5e7)+1, 10000)])

    # Draw divergence times
    tau1 = np.random.gamma(10.0, 1.0 / 10.0)
    tau2 = np.random.gamma(10.0, 1.0 / 10.0) + tau1
    tau3 = np.random.gamma(20.0, 1.0 / 10.0) + tau2

    # Draw mutation and recombination rates
    mu = np.random.uniform(2.5e-8, 2.5e-9)
    r  = np.random.uniform(2.5e-8, 2.5e-9)

    # Set up population configurations
    population_configurations = [
        msp.PopulationConfiguration(
            sample_size=5,
            initial_size=1000
        ),
        msp.PopulationConfiguration(
            sample_size=5,
            initial_size=1000
        ),
        msp.PopulationConfiguration(
            sample_size=5,
            initial_size=1000
        ),
        msp.PopulationConfiguration(
            sample_size=5,
            initial_size=1000
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

    return ([time_units, L, tau1, tau2, tau3, mu, r],
            msp.simulate(length=L, mutation_rate=mu, recombination_rate=r,
                         population_configurations=population_configurations,
                         demographic_events=demographic_events))

def hybrid_speciation(coal_units=1.0):
    """
    Out    P3    P2    P1
     |      |     |     |
     |      |     |     |
     |      |----/ \----|
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
    time_units = 2.0 * 1000.0 * coal_units

    # Draw sequence length
    L = np.random.choice([x for x in range(int(1e7), int(5e7)+1, 10000)])

    # Draw divergence times
    tau1 = np.random.gamma(10.0, 1.0 / 10.0)
    tau2 = np.random.gamma(10.0, 1.0 / 10.0) + tau1
    tau3 = np.random.gamma(20.0, 1.0 / 10.0) + tau2

    # Draw mutation and recombination rates
    mu = np.random.uniform(2.5e-8, 2.5e-9)
    r  = np.random.uniform(2.5e-8, 2.5e-9)

    # Draw hybridization fraction
    gamma = np.random.uniform(0.25, 0.75)

    # Set up population configurations
    population_configurations = [
        msp.PopulationConfiguration(
            sample_size=5,
            initial_size=1000
        ),
        msp.PopulationConfiguration(
            sample_size=5,
            initial_size=1000
        ),
        msp.PopulationConfiguration(
            sample_size=5,
            initial_size=1000
        ),
        msp.PopulationConfiguration(
            sample_size=5,
            initial_size=1000
        )
    ]

    # Set up demographic events
    demographic_events = [
        msp.MassMigration(
            time = tau1 * time_units,
            source = 1, destination = 2,
            proportion = gamma
        ),
        msp.MassMigration(
            time = tau1 * time_units + 1e-6,
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

    return ([time_units, L, tau1, tau2, tau3, mu, r, gamma],
            msp.simulate(length=L, mutation_rate=mu, recombination_rate=r,
                         population_configurations=population_configurations,
                         demographic_events=demographic_events))

def admixture(coal_units=1.0):
    """
    Out    P3    P2    P1
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
    time_units = 2.0 * 1000.0 * coal_units

    # Draw sequence length
    L = np.random.choice([x for x in range(int(1e7), int(5e7)+1, 10000)])

    # Draw divergence times
    tau1 = np.random.gamma(10.0, 1.0 / 10.0)
    tau2 = np.random.gamma(10.0, 1.0 / 10.0) + tau1
    tau3 = np.random.gamma(20.0, 1.0 / 10.0) + tau2

    # Draw mutation and recombination rates
    mu = np.random.uniform(2.5e-8, 2.5e-9)
    r  = np.random.uniform(2.5e-8, 2.5e-9)

    # Draw admixture proportion
    gamma = np.random.uniform(0.01, 0.25)

    # Draw admixture time, restrict to occurring between
    # and 10% and 90% of tau1
    admix_time = np.random.uniform(0.1 * tau1, 0.9 * tau1)

    # Set up population configurations
    population_configurations = [
        msp.PopulationConfiguration(
            sample_size=5,
            initial_size=1000
        ),
        msp.PopulationConfiguration(
            sample_size=5,
            initial_size=1000
        ),
        msp.PopulationConfiguration(
            sample_size=5,
            initial_size=1000
        ),
        msp.PopulationConfiguration(
            sample_size=5,
            initial_size=1000
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

    return ([time_units, L, tau1, tau2, tau3, mu, r, gamma, admix_time],
            msp.simulate(length=L, mutation_rate=mu, recombination_rate=r,
                         population_configurations=population_configurations,
                         demographic_events=demographic_events))

def admixture_w_gflow(coal_units=1.0):
    """
    Out    P3    P2    P1
     |      |     |<--->|
     |      |---->|<--->|
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
    time_units = 2.0 * 1000.0 * coal_units

    # Draw sequence length
    L = np.random.choice([x for x in range(int(1e7), int(5e7)+1, 10000)])

    # Draw divergence times
    tau1 = np.random.gamma(10.0, 1.0 / 10.0)
    tau2 = np.random.gamma(10.0, 1.0 / 10.0) + tau1
    tau3 = np.random.gamma(20.0, 1.0 / 10.0) + tau2

    # Draw mutation and recombination rates
    mu = np.random.uniform(2.5e-8, 2.5e-9)
    r  = np.random.uniform(2.5e-8, 2.5e-9)

    # Draw admixture proportion
    gamma = np.random.uniform(0.01, 0.25)

    # Draw admixture time, restrict to occurring between
    # and 10% and 90% of tau1
    admix_time = np.random.uniform(0.1 * tau1, 0.9 * tau1)

    m = np.random.uniform(2.5e-4, 5.0e-4)

    migration_matrix = [
        [0, m, 0, 0],
        [m, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    # Set up population configurations
    population_configurations = [
        msp.PopulationConfiguration(
            sample_size=5,
            initial_size=1000
        ),
        msp.PopulationConfiguration(
            sample_size=5,
            initial_size=1000
        ),
        msp.PopulationConfiguration(
            sample_size=5,
            initial_size=1000
        ),
        msp.PopulationConfiguration(
            sample_size=5,
            initial_size=1000
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

    return ([time_units, L, tau1, tau2, tau3, mu, r, gamma, admix_time, m],
            msp.simulate(length=L, mutation_rate=mu, recombination_rate=r,
                         population_configurations=population_configurations,
                         migration_matrix=migration_matrix,
                         demographic_events=demographic_events))
