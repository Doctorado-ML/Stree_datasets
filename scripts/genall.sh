#!/bin/bash
if [ "$1" = "" ] ; then
    echo "Hay que especificar si es pbs o slurm"
	exit 1
fi
if [[ ! "pbsslurm"  == *$1* ]] ; then
	echo "Hay que especificar si es pbs o slurm"
	exit 1
fi
for i in gridsearch gridbest cross; do
	echo "*** Building $i experiments"
	for j in stree odte bagging adaBoost; do
        for k in linear poly rbf any; do
		    ./genjobs.sh $i $j $k $1
        done
	done
done