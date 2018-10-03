#!/bin/bash


# FIX sswe/saved_models
mv dl4nlt/models/sswe/saved_models dl4nlt/models/sswe/local_mispelled

mkdir dl4nlt/models/sswe/saved_models/

mv dl4nlt/models/sswe/local_mispelled dl4nlt/models/sswe/saved_models/local_mispelled


# FIX lstm/saved_data
mv dl4nlt/models/lstm/saved_data dl4nltmodels/lstm/local_mispelled

mkdir dl4nlt/models/lstm/saved_data

mv dl4nlt/models/lstm/local_mispelled dl4nlt/models/lstm/saved_data/local_mispelled


# FIX lstm/runs
mv dl4nlt/models/lstm/runs dl4nltmodels/lstm/local_mispelled

mkdir dl4nlt/models/lstm/runs

mv dl4nlt/models/lstm/local_mispelled dl4nlt/models/lstm/runs/local_mispelled



