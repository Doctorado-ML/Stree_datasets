#!/bin/bash
for folder in gridsearch gridbest cross; do
    find $folder -type f -exec rm {} \;
done