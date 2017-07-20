'''
Asynchronous reward and punished thread respond to feedback in order to learn
target state:
    
    - T* back prop, frequency is TBD.
'''

# Imports 
import multiprocessing


class RewardPunishProcess(multiprocessing.Process):

    def __init__(self, _args):
        
        # init name
        self.process_name = "Reward Punish Process"
        
        # call super init
        super(RewardPunishProcess, self).__init__(name=self.process_name, args=_args)

    def run(self, *args):
        print 'In %s' % self.name
        return