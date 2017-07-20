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
    
    def __init__(self, eeg_feed_rate=250, epoch_duration=2, num_features=30):
        self.num_features = num_features
        self.cache_interval = int(eeg_feed_rate*epoch_duration)
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
        self.eeg_thread_event.set()
        thread = Thread(target=self.command_rx_thread)
        thread.start()
        
    def stop_eeg_thread(self):
        self.eeg_thread_event.clear()

    def eeg_rx_thread(self):
        '''
        Receiver will need to select correct stream, then continuously accept and 
        process commands as they arrive.
        '''
        
        rx_counter = 0
        while self.eeg_thread_event.isSet():
    
            # get command
            eeg_power, timestamps = self.inlet.pull_sample()
            
            assert len(eeg_power) == self.num_features
            
            # inc rx count
            rx_counter += 1
            
            # cache if apt.
            if rx_counter % self.cache_interval == 0:
                self.eeg_data_cache += [(eeg_power, timestamps)]
                
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    