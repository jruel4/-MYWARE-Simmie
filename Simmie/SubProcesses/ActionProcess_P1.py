'''
Forward Propogation of policy network. Synchronized to real-time user experience:
    
    - Policy net forward prop, locked to real-time action steps.
'''

# Imports 
import multiprocessing
import sys
from IPython.utils import io

class ActionProcess(multiprocessing.Process):
    
    def __init__(self, _args):
        
        self.logqueue = _args[2]

        # init name
        self.process_name = "Action Process"
        
        # call super init
        super(ActionProcess, self).__init__(name=self.process_name)

    def run(self, *args):
        self.logqueue.put('In %s' % self.name)
        return

