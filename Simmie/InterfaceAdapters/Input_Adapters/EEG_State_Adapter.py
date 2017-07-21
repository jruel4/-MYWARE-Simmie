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

- 2400 features per sample
'''

from threading import Thread, Event
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream
import numpy as np
from collections import deque

from Simmie.Simmie.InterfaceAdapters.Input_Adapters.Imprint_Adapter import ImprintAdapter
from Simmie.Simmie.InterfaceAdapters.Input_Adapters.Reward_Punish_Adapter import RewardPunishAdapter

class EEGStateAdapter:
    
    # n_timepoints 
    def __init__(self, n_freq=30, n_chan=8, eeg_feed_rate=250, samples_per_output=1, spectrogram_timespan=10, n_spectrogram_timepoints=10):
        self.num_channels = n_chan
        self.num_freqs = n_freq
        self.n_spectrogram_timepoints = n_spectrogram_timepoints
        self.eeg_fifo_len = spectrogram_timespan * eeg_feed_rate #assuming spectrogram_timespan is in seconds
        self.cache_interval = int(samples_per_output)
        self.eeg_thread_event = Event()
        self.eeg_data_cache = list()
        self.eeg_fifo = deque([[[0]*n_freq]*n_chan]*(eeg_feed_rate*n_spectrogram_timepoints), maxlen=self.eeg_fifo_len)
        
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
        
        
    def retrieve_latest_data(self):

        '''
        For V,F - T,F - V,B - PI,B
        '''
        if len(self.eeg_data_cache) > 0:
            
            # Sync timestamps
            self.sync_state_labels()
    
            # remove timestamps from eeg data
            eeg_data = np.asarray([d[0] for d in self.eeg_data_cache])
            
            # setup structure for tensorflow model
            data = eeg_data, np.asarray(self.rpv_data), np.asarray(self.imprint_data)
            
            # clear caches
            self.rpv_data = list()
            self.imprint_data = list()
            self.eeg_data_cache = list()
            
            return data
        else:
            return (None, None, None) 
    
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
        fifo_idx = np.linspace(0,self.eeg_fifo_len-1,self.num_spectrogram_timepoints).astype(int)
        while self.eeg_thread_event.isSet():
    
            # get command
            eeg_periodo, timestamp = self.inlet.pull_sample()
            
            assert len(eeg_periodo) == self.num_channels * self.num_freqs #lsl output is flattened periodo
            
            # add new periodogram to fifo
            self.eeg_fifo.append(eeg_periodo)
            
            # inc rx count
            rx_counter += 1
            
            # cache if apt.
            if rx_counter % self.cache_interval == 0:
                self.eeg_data_cache += [(np.asarray(self.eeg_fifo)[fifo_idx,:,:], timestamp)]
            
            
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    