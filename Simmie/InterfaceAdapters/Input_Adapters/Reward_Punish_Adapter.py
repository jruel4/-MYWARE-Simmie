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
    
    def __init__(self, p_keys=None, r_keys=None, dbg=True):
        self.thread_event = Event()
        self.rpv_data_cache = list()

        if p_keys == None:  self.p_keys = [ "0 pressed","NUMPAD0 pressed" ]
        else: self.p_keys = p_keys

        if r_keys == None:  self.r_keys = [ "1 pressed","NUMPAD1 pressed" ]
        else: self.r_keys = r_keys
        
        self.GOOD = (0,1)
        self.BAD = (1,0)
        
        self.dbg = dbg
            
    def get_data(self):
        data = self.rpv_data_cache
        self.rpv_data_cache = list()
        return data
    
    def launch_rpv_adapter(self, manual_stream_select=True): 
        print("Resolving RPV marker stream...")
        streams = resolve_stream('type', 'Markers')
        snum = 0
        if manual_stream_select:
            for i,s in enumerate(streams):
                print(i,s.name(), s.uid())
            snum = input("Select desired RewardPunish stream: ")
        self.inlet = StreamInlet(streams[int(snum)])
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
        
        output = (None,None) # one-hot output of good or bad
        while self.thread_event.isSet():
    
            # get command
            command_set, timestamps = self.inlet.pull_sample()
            
            
            if command_set[0] in self.r_keys:
                if self.dbg: print("GOOD")
                output = self.GOOD
            if command_set[0] in self.p_keys:
                if self.dbg: print("BAD")
                output = self.BAD
            else:
                continue
            # cache
            self.rpv_data_cache += [(output, timestamps)]
            
        print("Exiting imprint adapter rx thread")
        
        
        
if __name__ == "__main__":
    # Use 1/0 to print good/bad
    rpv = RewardPunishAdapter()
    rpv.launch_rpv_adapter()