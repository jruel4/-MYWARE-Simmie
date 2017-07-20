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

# Multiprocess debug setup 
multiprocessing.log_to_stderr(logging.DEBUG)
print("Cpu count: ",multiprocessing.cpu_count())

# Multiprocess management objects
manager = multiprocessing.Manager()
manager_dict = manager.dict()
event = multiprocessing.Event()

# Setup args
args = (manager_dict, event, )

# Init SubProcesses
process = ActionProcess(args)

# Declare daemon or not
process.daemon = False

# Start 
process.start()

# Join
process.join()

# Log complete
print("Complete")