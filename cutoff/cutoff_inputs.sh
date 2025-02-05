#!/bin/bash
 
# check the largest exponent in the basis and set the cutoff
# to be in the range 20*largest exponent to 80*largest exponent
# The rule of thumb is 50*largest exponent is a good value

cutoffs="150 200 250 300 350 400 450 500"
 
project=
input_file=$project.inp
 
for ii in $cutoffs ; do
    work_dir=cutoff_${ii}Ry
    if [ ! -d $work_dir ] ; then
        mkdir $work_dir
    else
        rm -r $work_dir/*
    fi
    #sed -e "s/LT_rel_cutoff/60/g" \
    # set both cutoff and relative curtoff to the same number
    sed -e "s/LT_cutoff/${ii}/g" \
        $input_file > $work_dir/$input_file
done
