#!/bin/bash
declare -a array=("1e3" "2e3" "5e3" "1e4" "2e4" "5e4" "1e5" "2e5" "5e5" "1e6" "2e6" "5e6" "1e7" "2e7" "5e7" "1e8" "1.5e8" "2e8" "2.5e8" "3e8" "4e8" "5e8")

# get length of an array
arraylength=${#array[@]}
p="qx"
filedir="..\/..\/getA\/32.50\/"

# use for loop to read all values and indexes
#for (( i=1; i<${arraylength}+1; i++ ));
#do
#for (( j=1; j<${arraylength}+1; j++ ));
#do
#  echo $i $j
#  sed -e "s/QX/${array[$i-1]}/g" -e "s/QY/${array[$j-1]}/g" -e "s/FILESDIR/$filedir/g" sstg.temp.py > sstg.$p.$i.$j.py
#  sed -e "s/QX/$i/g" -e "s/QY/$j/g" -e "s/P/$p/g" py.temp.slurm > sstg.$p.$i.$j.slurm
#  sbatch sstg.$p.$i.$j.slurm
#done
#done
for (( i=1; i<${arraylength}+1; i++ ));
do
  echo $i
  sed -e "s/QX/${array[$i-1]}/g" -e "s/FILESDIR/$filedir/g" sstg.temp.py > sstg.$p.$i.py
  sed -e "s/QX/$i/g" -e "s/P/$p/g" py.temp.slurm > sstg.$p.$i.slurm
  sbatch sstg.$p.$i.slurm
done
