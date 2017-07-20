'''
Source: User via KeyInput LSL App

Description: Positive, negative and neutral feedback from user for training Target network.

subprocesses: P4 input
'''

'''
Reward punish data will comm from LSL Key Capture App. 

Assume some format
'''

from threading import Thread, Event
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream

class RewardPunishAdapter:
    
    def __init__(self):
        self.thread_event = Event()
        self.rpv_data_cache = list()
        
    def get_data(self):
        data = self.rpv_data_cache
        self.rpv_data_cache = list()
        return data
    
    def launch_rpv_adapter(self, manual_stream_select=True): 
        streams = resolve_stream('type', 'AudioCommands')
        snum = 0
        if manual_stream_select:
            for i,s in enumerate(streams):
                print(i,s.name())
            snum = input("Select desired RewardPunish stream: ")
        self.inlet = StreamInlet(streams[snum])
        # launch thread
        self.thread_event.set()
        thread = Thread(target=self.command_rx_thread)
        thread.start()
        
    def stop_imprint_thread(self):
        self.thread_event.clear()

    def command_rx_thread(self):
        '''
        Receiver will need to select correct stream, then continuously accept and 
        process commands as they arrive.
        '''
        
        while self.thread_event.isSet():
    
            # get command
            command_set, timestamps = self.inlet.pull_sample()
            
            # cache
            self.rpv_data_cache += [(command_set, timestamps)]
            
        print("Exiting imprint adapter rx thread")
        
        
        
        
        
        
        
        
        




