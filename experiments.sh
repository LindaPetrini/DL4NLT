#!/bin/bash


python -m dl4nlt.models.sswe.train --dataset=global_mispelled

python -m dl4nlt.models.lstm.experiments
