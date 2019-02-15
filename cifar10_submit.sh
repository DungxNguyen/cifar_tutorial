#!/bin/bash
for dropout in 0.8 1.0 0.5
do
    for init_lr in 0.01 0.03 
    do
	for decay_lr in 0.1 0.5 
	do
	    for batch_size in 128
	    do
		qsub -v init_lr=$init_lr,use_momentum=True,momentum_alpha=0.9,dropout=$dropout,batch_size=$batch_size,decay_lr=$decay_lr cifar10.qsub
	    done
	done
    done

    for init_lr in 0.003
    do
	for momentum_alpha in 0.99
	do
	    for batch_size in 128
	    do
		qsub -v init_lr=$init_lr,use_momentum=True,momentum_alpha=$momentum_alpha,dropout=$dropout,batch_size=$batch_size,decay_lr=0.1 cifar10.qsub
	    done
	done
    done
done
