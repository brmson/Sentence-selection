#!/bin/bash
#
# A script to convert a set of (question, sentence, binLabel) tuples
# (where sentence may be a passage or, with -p, property label)
# to a classifier which attempts to predict binLabel from unseen
# (question, sentences) pairs.
#
# Usage: std-run.sh [-p] TRAINDATAPATH
#
# Example: ./std-run.sh -p ../yodaqa/data/ml/embsel/propdata/

props=false
if [ "$1" = "-p" ]; then
	props=true
	shift
fi

path=$1
cp -a "$1." "data/curated-train"

# Convert YodaQA-generated data to Jacana-style data
if [[ props ]]
then
	echo 'Running property-reparse'
	python reparseprops.py
else
	echo 'Running sentence-reparse'
	python reparse.py
fi

# Convert Jacana-style data to pickled Python data structures
echo 'Running save.py'
python save.py

# Train and save a classifier on top of the pickled data
echo 'Running train.py'
python train.py
