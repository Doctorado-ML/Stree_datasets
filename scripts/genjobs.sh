#!/bin/bash
if [ "$1" = "" -o "$2" = "" -o "$3" = "" ] ; then
	echo "Hay que seleccionar:"
    echo " - el tipo de experimento {gridsearch, gridbest, cross}"
    echo " - el modelo {stree, adaBoost, bagging, odte}"
    echo " - el kernel {linear, poly, rbf, any}"
	exit 1
fi
if [[ ! "gridsearchgridbestcross"  == *$1* ]] ; then
	echo "Hay que seleccionar el tipo de experimento {gridsearch, gridbest, cross}"
	exit 1
fi
if [[ ! "streeadaBoostbaggingodte"  == *$2* ]] ; then
	echo "Hay que seleccionar el modelo {stree, adaBoost, bagging, odte}"
	exit 1
fi
if [[ ! "linearpolyrbfany"  == *$3* ]] ; then
	echo "Hay que seleccionar el kernel {linear, poly, rbf, any}"
	exit 1
fi
script_path="$(pwd)/.."
cp experiment.template experiment_$1.sh
perl -i -pe"s/<model>/$2/g" experiment_$1.sh
perl -i -pe"s~<folder>~$script_path~g" experiment_$1.sh
perl -i -pe"s/<experiment>/$1/g" experiment_$1.sh
mkdir -p $1/$2/$3
cat ../datasets.txt|cut -d " " -f 2|tail -49|while read a; do
	cp experiment_$1.sh $1/$2/$3/experiment_$a.sh
	perl -i -pe"s/<data>/$a/g" $1/$2/$3/experiment_$a.sh
	perl -i -pe"s/<kernel>/$3/g" $1/$2/$3/experiment_$a.sh
done
rm experiment_$1.sh