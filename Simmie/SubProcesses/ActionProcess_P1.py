'''
Forward Propogation of policy network. Synchronized to real-time user experience:
    
    - Policy net forward prop, locked to real-time action steps.
'''

# Imports 
import multiprocessing


class ActionProcess(multiprocessing.Process):
    
    def __init__(self, _args):
        
        # init name
        self.process_name = "Action Process"
        
        # call super init
        super(ActionProcess, self).__init__(name=self.process_name, args=_args)

    def run(self, *args):
        print 'In %s' % self.name
        return

