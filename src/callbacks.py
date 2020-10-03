import numpy as np
import tensorflow as tf


def early_stopping(loss_list, min_delta=0.1, patience=20):
    # No early stopping for 2*patience epochs
    if len(loss_list) // patience < 2:
        return False
    # Mean loss for last patience epochs and second-last patience epochs
    mean_previous = np.mean(loss_list[::-1][patience:2 * patience])  # second-last
    mean_recent = np.mean(loss_list[::-1][:patience])  # last
    # you can use relative or absolute change
    delta_abs = np.abs(mean_recent - mean_previous)  # abs change
    delta_abs = np.abs(delta_abs / mean_previous)  # relative change
    if delta_abs < min_delta:
        print("*CB_ES* Loss didn't change much from last %d epochs" % (patience))
        print("*CB_ES* Percent change in loss value:", delta_abs * 1e2)
        return True
    else:
        return False
