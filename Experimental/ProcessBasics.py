#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 00:02:43 2017

@author: marzipan
"""


        
        
import multiprocessing
import logging

def worker(num):
    """thread worker function"""
    print 'Worker:', num
    return

if __name__ == '__main__':
    multiprocessing.log_to_stderr(logging.DEBUG) # debugginh
    jobs = []
    # TODO pass Event as argument!
    for i in range(5):
        p = multiprocessing.Process(name='my_service', target=worker, args=(i,))
        p.daemon = False # set daemon=True so let parnents exit without waiting for child
        jobs.append(p)
        p.start()
        
        
# wrap top level in main to avoid recursion
import multiprocessing
import multiprocessing_import_worker

if __name__ == '__main__':
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=multiprocessing_import_worker.worker)
        jobs.append(p)
        p.start()
        
        
# Create worker as subclass
class Worker(multiprocessing.Process):

    def run(self):
        print 'In %s' % self.name
        return

if __name__ == '__main__':
    jobs = []
    for i in range(5):
        p = Worker()
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()
        
# Condition object for synchronizing
import multiprocessing
import time

def stage_1(cond):
    """perform first stage of work, then notify stage_2 to continue"""
    name = multiprocessing.current_process().name
    print 'Starting', name
    with cond:
        print '%s done and ready for stage 2' % name
        cond.notify_all()

def stage_2(cond):
    """wait for the condition telling us stage_1 is done"""
    name = multiprocessing.current_process().name
    print 'Starting', name
    with cond:
        cond.wait()
        print '%s running' % name

if __name__ == '__main__':
    condition = multiprocessing.Condition()
    s1 = multiprocessing.Process(name='s1', target=stage_1, args=(condition,))
    s2_clients = [
        multiprocessing.Process(name='stage_2[%d]' % i, target=stage_2, args=(condition,))
        for i in range(1, 3)
        ]

    for c in s2_clients:
        c.start()
        time.sleep(1)
    s1.start()

    s1.join()
    for c in s2_clients:
        c.join()
        
        
# shared data with manager
import multiprocessing

def worker(d, key, value):
    d[key] = value

if __name__ == '__main__':
    mgr = multiprocessing.Manager()
    d = mgr.dict()
    jobs = [ multiprocessing.Process(target=worker, args=(d, i, i*2))
             for i in range(10) 
             ]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    print 'Results:', d
    
# Cpu data access
pool_size = multiprocessing.cpu_count() * 2