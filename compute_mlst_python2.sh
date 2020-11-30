###===================
#!/bin/bash
#PBS -l select=1:ncpus=14:mem=84gb:pcmem=6gb -l walltime=2:00:00
#PBS -l cput=28:00:00
#PBS -q high_pri
#PBS -W group_list=mstrout
###-------------------

echo "Node name:"
hostname

cd /xdisk/kobourov/mig2020/extra/abureyanahmed/mlst_kruskal
module load python/3.5/3.5.5
module load matlab/r2018b
python3 run_mlst_python2.py $MAP_FILE $PBS_ARRAY_INDEX > $LOG_FOLDER/output_$PBS_ARRAY_INDEX.dat 2> $LOG_FOLDER/error_$PBS_ARRAY_INDEX.dat

