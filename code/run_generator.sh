#!/usr/bin/env bash

# Number of supernodes and superedges
Ms=(10 20 40 100 200)
# Node and hedge cluster size
ps=(5 10 20)
# Noise level
es=(0 0.2 0.25 0.3)
# path to directory for output files
base_path='../data/randg/'
# seed
seed=0

mkdir -p ${base_path}

for M in "${Ms[@]}"; do
    for p in "${ps[@]}"; do
        for e in "${es[@]}"; do
            echo "---- `date`"
            python run_random_gen.py -M $M -p $p -r $p -e $e -out_dir ${base_path} -seed ${seed}
        done
    done
done
