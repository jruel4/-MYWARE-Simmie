'''
Launches / Manages Subprocesses
'''

# Internal Imports
from Simmie.SubProcesses.ActionProcess_P1 import ActionProcess
from Simmie.SubProcesses.ImprintProcess_P2 import ImprintProcess
from Simmie.SubProcesses.BiometricValueProcess_P3 import BiometricValueProcess
from Simmie.SubProcesses.RewardPunishProcess_P4 import RewardPunishProcess

# External Imports
import multiprocessing
import logging
import sys

# Multiprocess debug setup 
multiprocessing.log_to_stderr(logging.DEBUG)
print("Cpu count: ",multiprocessing.cpu_count())

if __name__ == "__main__":
    
    # Multiprocess management objects
    manager = multiprocessing.Manager()
    manager_dict = manager.dict()
    event = multiprocessing.Event()
    logqueue = multiprocessing.Queue()

    # Setup args
    args = (manager_dict, event, logqueue, )
    
    # Init SubProcesses
    action_process = ActionProcess(args)
    imprint_process = ImprintProcess(args)
    biometric_value_process = BiometricValueProcess(args)
    reward_punish_process = RewardPunishProcess(args)
    
    # Collect subprocesses into dict for easy manipulation
    subprocess_dict = {'action_process': action_process,
                       'imprint_process': imprint_process,
                       'biometric_value_process': biometric_value_process,
                       'reward_punish_process': reward_punish_process}
    
    # Declare daemon or not
    for p in subprocess_dict.values():
        p.daemon = False
    
    # Start 
    for p in subprocess_dict.values():
        p.start()
    
    # Join
    for p in subprocess_dict.values():
        p.join()

    print(logqueue.get())
    
    # Log complete
    print("Complete")
    


















