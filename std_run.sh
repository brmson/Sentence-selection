#!/bin/bash
props=false
if [ "$1" = "-p" ]; then
props=true
shift; 
fi
path=$1
cp -a "$1." "data/curated-train"
if [[ props ]]
then
echo 'Running property-reparse'
python reparseprops.py
else
echo 'Running sentence-reparse'
python reparse.py
fi
echo 'Running save.py'
python save.py
echo 'Running train.py'
python train.py
