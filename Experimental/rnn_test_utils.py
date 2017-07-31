#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 21:30:40 2017

@author: marzipan
"""

def pretty_print_output(outputs, states):
    '''
    [batch_size, max_time, cell.output_size]
    
    states.c == (1,7) max_time X num_units
    states.h == (1,7) max_time X num_units
    '''
    
    print("Outputs:\n")
    print(str(outputs)+"\n")
    for n,state in enumerate(states):
        print("Cell "+str(n)+" memory: \n")
        print(str(state.c)+"\n")
        print("Cell "+str(n)+" hidden: \n")
        print(str(state.h)+"\n")
        
        