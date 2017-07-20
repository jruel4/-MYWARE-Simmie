'''
Source: MartianBCI Pipeline

Description: Formatted EEG data of basic and expert features.

subprocesses: P1 input and P2 input
'''

'''
EEG features will just be spectrogram at first.

- 5 time points (10 seconds w/ 2 second epochs)
- 30 frequencies
- 16 channels
'''

from threading import Thread, Event
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream

class EEGStateAdapter:
    
    def __init__(self):
        self.eeg_thread_event = Event()
        self.eeg_data_cache = list()
        
    def retrieve_eeg_state(self):
        #TODO what numpy format is good?
        pass
    
    def launch_eeg_adapter(self, manual_stream_select=True): 
        streams = resolve_stream('type', 'AudioCommands')
        snum = 0
        if manual_stream_select:
            for i,s in enumerate(streams):
                print(i,s.name())
            snum = input("Select EEGStateAdapter stream: ")
        self.inlet = StreamInlet(streams[snum])
        # launch thread
        self.command_thread_event.set()
        thread = Thread(target=self.command_rx_thread)
        thread.start()
        
    def stop_eeg_thread(self):
        self.command_thread_event.clear()

    def eeg_rx_thread(self):
        '''
        Receiver will need to select correct stream, then continuously accept and 
        process commands as they arrive.
        '''
        
        while self.command_thread_event.isSet():
    
            # get command
            command_set, timestamps = self.inlet.pull_sample()
            
            # cache
            self.imprint_data_cache += [(command_set, timestamps)]
            
        print("Exiting imprint adapter rx thread")
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    