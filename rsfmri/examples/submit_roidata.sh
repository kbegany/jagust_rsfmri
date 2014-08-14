#!/bin/bash
#
# This simple SGE batch script shows how to run a python script
# Below are SGE options embedded in comments
#

# specify number of jobs to run (1-N)
# eg run 4 subjects  -t 1-4
#$ -t 1-192

# join stdout and stderr
# this may make debugging easier, but output may become less
# readable.

#$ -j y

# redefine output file
# this saves the output from your cluster jobs to your home directory
# and names the files <scriptname>_<node=cn1-cn15>_<jobnumber>
#    eg. python_wrapper.sh_cn12.7535
#$ -o /home/jagust/graph/scripts/sge_scripts/logs/$JOB_NAME_$HOSTNAME.$JOB_ID

# Shell to use: Specifies the interpreting shell for the job

#$ -S /bin/bash

# Specifies that all environment variables active within the  qsub
#     utility be exported to the context of the job.
#$ -V

# Tells SGE to send a mail to the job owner when
#   the job begins (b), ends (e), aborted (a), and suspended(s).
#$ -m as

# Tells SGE to send mail about this job to the given email address
# -M youremailaddress@berkeley.edu

# Run python program
which python
export PYTHONPATH=/home/jagust/cindeem/CODE/jagust_rsfmri:/usr/local/anaconda/pkgs
# run despiked spm data
#python /home/jagust/cindeem/CODE/jagust_rsfmri/rsfmri/examples/roi_data.py /home/jagust/graph/data/mri1.5/tr220  -d -spm
#python /home/jagust/cindeem/CODE/jagust_rsfmri/rsfmri/examples/roi_data.py /home/jagust/graph/data/mri1.5/tr220  -d -spm -gsr
# run despiked ANTS data
/usr/local/anaconda/bin/python /home/jagust/cindeem/CODE/jagust_rsfmri/rsfmri/examples/roi_data.py /home/jagust/graph/data/mri1.5/tr220  -d
#/usr/local/anaconda/bin/python /home/jagust/cindeem/CODE/jagust_rsfmri/rsfmri/examples/roi_data.py /home/jagust/graph/data/mri1.5/tr220  -d -gsr
# run non-despiked spm data
#python /home/jagust/cindeem/CODE/jagust_rsfmri/rsfmri/examples/roi_data.py /home/jagust/graph/data/mri1.5/tr220  -spm
#python /home/jagust/cindeem/CODE/jagust_rsfmri/rsfmri/examples/roi_data.py /home/jagust/graph/data/mri1.5/tr220  -spm -gsr
# run non-despiked ANTS data
#python /home/jagust/cindeem/CODE/jagust_rsfmri/rsfmri/examples/roi_data.py /home/jagust/graph/data/mri1.5/tr220
#python /home/jagust/cindeem/CODE/jagust_rsfmri/rsfmri/examples/roi_data.py /home/jagust/graph/data/mri1.5/tr220  -gsr


## connectome
#python /home/jagust/cindeem/CODE/jagust_rsfmri/rsfmri/examples/roi_data.py /home/jagust/graphlbl/connectome_data/conn4graph  -d
#python /home/jagust/cindeem/CODE/jagust_rsfmri/rsfmri/examples/roi_data.py /home/jagust/graphlbl/connectome_data/conn4graph -d -gsr



