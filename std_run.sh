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

if [[ -f resources/glove.6B.50d.txt ]]
then
echo "Dictionary allready downloaded"
else
echo "Downloading dictionary"
wget http://pasky.or.cz/dev/brmson/glove.6B.50d.txt.gz
gunzip glove.6B.50d.txt.gz
mkdir -p resources
mv glove.6B.50d.txt resources
fi



props=false
if [ "$1" = "-p" ]; then
	props=true
	shift
fi
path=$1

# Convert YodaQA-generated data to Jacana-style data
if [[ props ]]
then
	echo 'Running property-reparse'
	python reparseprops.py "$path"
else
	echo 'Running sentence-reparse'
	python reparse.py "$path"
fi

# Convert Jacana-style data to pickled Python data structures
echo 'Running save.py'
python save.py

# Train and save a classifier on top of the pickled data
echo 'Running train.py'
python train.py
