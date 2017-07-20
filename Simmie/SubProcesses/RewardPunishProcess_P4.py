'''
Asynchronous reward and punished thread respond to feedback in order to learn
target state:
    
    - T* back prop, frequency is TBD.
'''

# Imports 
import multiprocessing


class RewardPunishProcess(multiprocessing.Process):

    def run(self):
        print 'In %s' % self.name
        return

