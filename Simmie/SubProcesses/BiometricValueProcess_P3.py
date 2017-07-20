'''
Asynchronous BV thread should run:
    
    - V* forward prop (predict return of current state)
    
    - T* foward prop (calculate actual reward of new state)
    
    - V* back prop (learn from TD error)
'''

# Imports 
import multiprocessing


class BiometricValueProcess(multiprocessing.Process):
    
    def __init__(self, _args):
        
        # init name
        self.process_name = "Biometric Value Process"
        
        # call super init
        super(BiometricValueProcess, self).__init__(name=self.process_name, args=_args)

    def run(self, *args):
        print 'In %s' % self.name
        return