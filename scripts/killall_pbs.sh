#!/bin/bash
qstat -u Ricardo.Montanana|tail -n +6|cut -b -6|while read job; do qdel $job; done