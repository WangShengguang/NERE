#!/usr/bin/env bash
python3 manage.py --ner all --mode train  &>ner_train.out
python3 manage.py --re all --mode train  &>re_train.out
python3 manage.py --joint --mode train  &>joint_train.out