#!/bin/bash
if [ "$1" = "" -o "$2" = "" -o "$3" = "" -o "$4" = "" -o "$5" = "" ] ; then
	echo "Hay que seleccionar:"
    echo " - el tipo de experimento {gridsearch, gridbest, cross}"
    echo " - el modelo {stree, adaBoost, bagging, odte}"
    echo " - el kernel {linear, poly, rbf, any}"
	echo " - el tipo de plataforma {pbs, slurm}"
	echo " - el archivo con nombres de datasets"
	echo "opcionalmente al final: dry-run"
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
if [[ ! "pbsslurm"  == *$4* ]] ; then
	echo "Hay que especificar la plataforma {pbs, slurm}"
	exit 1
fi
if [ $4 = "pbs" ] ; then
	launcher="qsub"
else
	launcher="sbatch"
fi
script_path="$(pwd)"
cd $1/$2/$3
counter=0
lines="$(cat $script_path/$5|cut -d " " -f 2|tail -49)"
for a in $lines; do
	echo "launch experiment_$a.sh"
	if [ "$6" = "dry-run" ] ; then
		echo "not launched"
	else
		$launcher experiment_$a.sh
	fi
	let counter++
done
if [ "$6" = "dry-run" ] ; then
	echo "Not launched $counter jobs"
else
	echo "Launched $counter jobs"
fi