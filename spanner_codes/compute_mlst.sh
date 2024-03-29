###===================
#!/bin/bash
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=6gb -l walltime=4:00:00
#PBS -l cput=112:00:00
#PBS -q standard
#PBS -W group_list=mstrout
###-------------------

echo "Node name:"
hostname

cd /groups/kobourov/abureyanahmed/Graph_spanners
module load python/3.5/3.5.5
module load matlab/r2018b
python3 run_mlst.py $MAP_FILE $PBS_ARRAY_INDEX > $LOG_FOLDER/output_$PBS_ARRAY_INDEX.dat 2> $LOG_FOLDER/error_$PBS_ARRAY_INDEX.dat

