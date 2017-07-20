'''
Asynchronous BV thread should run:
    
    - V* forward prop (predict return of current state)
    
    - T* foward prop (calculate actual reward of new state)
    
    - V* back prop (learn from TD error)
'''

# Imports 
import multiprocessing


class BiometricValueProcess(multiprocessing.Process):

    def run(self):
        print 'In %s' % self.name
        return

