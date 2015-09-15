#!/bin/bash
if [[ $1=='props' ]]
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
