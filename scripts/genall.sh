#!/bin/bash
for i in gridsearch gridbest cross; do
	echo "*** Building $i experiments"
	for j in stree odte bagging adaBoost; do
        for k in linear poly rbf any; do
		    ./genjobs.sh $i $j $k
        done
	done
done