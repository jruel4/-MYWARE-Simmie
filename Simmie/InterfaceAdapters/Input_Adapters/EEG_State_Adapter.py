'''
Source: MartianBCI Pipeline

Description: Formatted EEG data of basic and expert features.

subprocesses: All backprops utilize correctly time-associated EEG data.
'''

'''
EEG features will just be spectrogram at first.

- 5 time points (10 seconds w/ 2 second epochs)
- 30 frequencies
- 16 channels
'''

from threading import Thread, Event
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream
import numpy as np

from Imprint_Adapter import ImprintAdapter
from Reward_Punish_Adapter import RewardPunishAdapter

#TODO when to call sync? how much data to cache? its synchronous so we can be consistent right?
#TODO decide when to clear eeg_data_cache

class EEGStateAdapter:
    
    def __init__(self, eeg_feed_rate=250, epoch_duration=2, num_features=30):
        self.num_features = num_features
        self.cache_interval = int(eeg_feed_rate*epoch_duration)
        self.eeg_thread_event = Event()
        self.eeg_data_cache = list()
        
        # Init other adapters
        self.imprintAdapter = ImprintAdapter()
        self.rpvAdapter = RewardPunishAdapter()
        
        self.rpvDataSynced = list()
        self.imprintDataSynced = list()
        
    def sync_state_labels(self):
        '''
        Grab all available labels data from imprint and rpv adapters and sync 
        timestamps.
        '''
        '''
        syncing data is basically a matter of comparing timestamps and associating the 
        closest ones. This should be easy if we are storing lots of data...
        
        we can collect it all to be able to associate the closest, then we can 
        clear cache after a second or so.
        '''
        
        # Sync rpv data
        self.rpv_data,ts_diffs = self.sync_data( self.eeg_data_cache,  self.rpvAdapter.get_data())
        
        # Sync Imprint data
        self.imprint_data,ts_diffs = self.sync_data( self.eeg_data_cache,  self.imprintAdapter.get_data())  
        
    def retrieve_training_data(self):
        '''
        For training Policy and Target networks
        '''
        training_data = np.asarray(self.rpv_data), np.asarray(self.imprint_data)
        self.rpv_data = list()
        self.imprint_data = list()
        return training_data
        
        
    def retrieve_latest_eeg_state(self):
        '''
        For V,F - T,F - V,B - PI,B
        '''
        if len(self.eeg_data_cache) > 0:
            latest_state, timestamp = self.eeg_data_cache.pop()
            return np.asarray(latest_state)
        else:
            return False 
    
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
            
            
    def sync_data(self, _inputs, labels):
        '''
        assume inputs and labels both lists of tuples with first val value, second val timestamp.
        
        assume we have many more inputs than labels
        
        strategy is to search for the closest label for each input, also we can only,
        use one input per label so if the next closest is further than a removed one 
        then we would like to collect statistics on that.
        
        '''   
        
        ts_diffs = []    
        synced_pairs = []
                 
        # copy inputs
        inputs = list(_inputs)
        
        # loop over labels to find closest input
        for label in labels:
            
            # extract timestamps to array
            inputs_ts = np.asarray([i[1] for i in inputs])
            
            # extract label ts
            label_ts = label[1]
            
            # find nearest input to label
            inputs_ts = abs( inputs_ts - label_ts )
            amin = np.argmin(inputs_ts)
            closest_input = inputs[amin][0]
            
            # collect metrics
            ts_diffs += [inputs_ts[amin]]
            
            # add pair
            synced_pairs += [(closest_input, label[0])] 
            
            # delete used input
            del inputs[amin]
            
        return synced_pairs, ts_diffs
        
#==============================================================================
# if __name__ == '__main__':
#     
#     # create fake input and lable data
#     fake_input = [(i+.01,i+.02) for i in range(10)] + [(666,0),(666,4.0)]      
#     fake_label = [(i,i) for i in range(10)]
#     
#     print(sync_data(fake_input, fake_label))
#         
#     
#==============================================================================
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    