#!/bin/bash

# qsub -cwd -t 1-45:1\
qsub -cwd -t 46-49:1\
    -N svols -l h_data=4G,h_rt=00:30:00 -pe shared 2 -o logs -j y \
    ./scripts/array_job.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10}
