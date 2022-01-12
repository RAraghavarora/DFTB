#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --partition=gpu2                        # specify ml partition or gpu2 partition
#SBATCH --gres=gpu:4                      # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --nodes=1                        # request 1 node
#SBATCH --ntasks=8
#SBATCH -J fchl-ra
#SBATCH --output=dip.out
#SBATCH --error=dip.err
#SBATCH -A p_biomolecules
#SBATCH --mail-type=all
#SBATCH        --mail-user=leonardo.medrano@nano.tu-dresden.de
#SBATCH --mem-per-gpu=8000MB
ulimit -s unlimited
echo Starting Program
module purge                                 # purge if you already have modules loaded
module load modenv/scs5
module load Python/3.6.4-intel-2018a
. /home/medranos/vdftb20/bin/activate
module load cuDNN/8.0.4.30-CUDA-11.1.1
echo "training starts"
walltime=$(squeue -h -j $SLURM_JOBID -o "%L")
IFS=- read daysleft rest <<< "$walltime"
if [ -z "$rest" -a "$rest" != " " ]; then
    rest=$daysleft
    daysleft=0
fi
IFS=: read hsleft minsleft secsleft <<< "$rest"
hslefttot=$(($daysleft*24 + $hsleft))
walltime1=$(date -u -d "$rest" +"%H:%M:%S")
walltime2=$daysleft" days "$(date -u -d "$rest" +"%H hours %M minutes %S seconds")
echo "*** JOB '"$SLURM_JOB_NAME"' (ID: "$SLURM_JOBID") ***"
echo "*** "$SLURM_NODELIST": "$SLURM_JOB_NUM_NODES" node(s),  "$SLURM_NTASKS" core(s) in total ***"
echo "*** Submitted in: ${SLURM_SUBMIT_DIR} ***"
echo ""
echo "*** [TIMING] start "$(date "+%b %d, %H:%M:%S")" ***"
echo "*** [TIMING] walltime "$walltime" ["$hslefttot":"$minsleft":"$secsleft"] ends "$(date -d "$walltime2" "+%b %d, %H:%M:%S")" ***"
echo ""
echo ""
export OMP_NUM_THREADS=1
echo ""
echo "JOB OUTPUT:"
echo "###########################################################################"
echo ""
SECONDS=0
echo "training starts"
#unset I_MPI_PMI_LIBRARY
#export DFTB_COMMAND='mpiexec -n 1 /home/medranos/vdftb20/dftb/bin/dftb+'
#export DFTB_PREFIX='/home/medranos/SK-files/3ob-3-1/'

work=/scratch/ws/1/medranos-DFTB/raghav/code
python3 $work/schnet/train_fchl.py

echo "training is over :-)"
EXTSTAT=$?
echo ""
echo "###########################################################################"
echo ""
echo "*** [TIMING] end "$(date "+%b %d, %H:%M:%S")" ***"
echo "*** [TIMING] duration "$(squeue -h -j $SLURM_JOBID -o "%M")"["$(($SECONDS/3600))":"$(TZ=UTC0 printf '%(%M:%S)T\n' $SECONDS)"] ***"
echo ""
exit $EXTSTAT
