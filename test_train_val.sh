#! /bin/bash

set -euxETo pipefail

WORKDIR="traces_out"
DATASET="malware_web_mixed_cleaned.csv"

malware_web_TRAIN_SIZE=200000
TEST_SIZE=90000

# change to the working directory
cd $WORKDIR

# shuffle the dataset
MEMORY=200 ../../terashuf/terashuf < $DATASET > ${DATASET}_shuf.csv

# first split the train set
split -l $malware_web_TRAIN_SIZE ${DATASET}_shuf.csv malware_web_train_
mv malware_web_train_aa malware_web_train.csv

# cat the rest of the files together
cat malware_web_train_* > tmp

# split the test set
split -l $TEST_SIZE tmp malware_web_test_
mv malware_web_test_aa malware_web_test.csv

# cat the rest of the files to get the val set
cat malware_web_test_* > malware_web_val.csv

# remove the temporary files
rm tmp
rm malware_web_train_*
rm malware_web_test_*