#!/bin/bash

git clone https://github.com/chanzuckerberg/MedMentions.git
mkdir -p data/medmentions_st21pv
cp MedMentions/st21pv/data/corpus_pubtator.txt.gz data/medmentions_st21pv/
gunzip data/medmentions_st21pv/corpus_pubtator.txt.gz


rm -rf MedMentions