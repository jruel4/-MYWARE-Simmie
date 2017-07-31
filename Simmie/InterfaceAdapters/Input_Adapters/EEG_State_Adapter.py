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

from Simmie.InterfaceAdapters.Input_Adapters.Imprint_Adapter import ImprintAdapter
from Simmie.InterfaceAdapters.Input_Adapters.Reward_Punish_Adapter import RewardPunishAdapter

class EEGStateAdapter:
    
    # n_timepoints 
    def __init__(self, n_freq=30, n_chan=8, eeg_feed_rate=250, samples_per_output=1, spectrogram_timespan=10, n_spectrogram_timepoints=10):
        self.num_channels = n_chan
        self.num_freqs = n_freq
        self.n_spectrogram_timepoints = n_spectrogram_timepoints
        self.eeg_fifo_len = spectrogram_timespan * eeg_feed_rate #assuming spectrogram_timespan is in seconds
        
        # Verify this is an int, then cast to int
        assert self.eeg_fifo_len.is_integer(), "Spectrogram timespan (" + str(spectrogram_timespan) +") * SPS (" + str(eeg_feed_rate) + ") must be an integer, is: " + str(self.eeg_fifo_len)
        self.eeg_fifo_len = int(self.eeg_fifo_len)

        self.cache_interval = int(samples_per_output)
        self.eeg_thread_event = Event()
        self.eeg_data_cache = list()
        self.eeg_fifo = deque([], maxlen=self.eeg_fifo_len)
        
        # Init other adapters
        self.imprintAdapter = ImprintAdapter(dbg=False)
        self.rpvAdapter = RewardPunishAdapter(dbg=False)
        
        self.rpv_data_dict = dict()
        self.imprint_data_dict = dict()
        
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
        self.rpv_data_dict = self.sync_data( self.eeg_data_cache,  self.rpvAdapter.get_data())
        
        # Sync Imprint data
        self.imprint_data_dict = self.sync_data( self.eeg_data_cache,  self.imprintAdapter.get_data())  
        
        
    def retrieve_latest_data(self):

        '''
        For V,F - T,F - V,B - PI,B
        '''
        if len(self.eeg_data_cache) > 0:

#==============================================================================
#             data = (
#              np.asarray([d[0] for d in self.eeg_data_cache]),
#              np.ones((1,1,10,240)), #np.empty(0),#[], #
#              np.asarray([(1,0)]),# np.empty(0),#[], #np.ones((1,1,10,240)),
#              np.ones((1,1,10,240)),#np.empty(0),#[], #np.ones((1,1,10,240)),
#              np.asarray([44])
#                      #np.empty(0)#[]
#              )
#             self.clear_caches()
#             return data
#==============================================================================

            # Sync timestamps
            self.sync_state_labels()
    
            # remove timestamps from eeg data
            eeg_data = np.asarray([d[0] for d in self.eeg_data_cache])
            
            # setup structure for tensorflow model
            
            rpv_inputs = np.asarray(self.rpv_data_dict['inputs'])
            rpv_labels = np.asarray(self.rpv_data_dict['labels'])
            
            assert rpv_inputs.shape[0] == rpv_labels.shape[0] #TODO error string here
            
            imp_inputs = np.asarray(self.imprint_data_dict['inputs'])
            imp_labels = np.asarray(self.imprint_data_dict['labels'])    #TODO error
            
            assert imp_inputs.shape[0] == imp_labels.shape[0]
            
            data = eeg_data, rpv_inputs, rpv_labels, imp_inputs, imp_labels

            # clear caches
            self.clear_caches()
            
            return data
        else:
            return ([], [], [], [], []) 
    
    def launch_eeg_adapter(self, manual_stream_select=True):
        
        self.imprintAdapter.launch_imprint_adapter()
        self.rpvAdapter.launch_rpv_adapter()

        print("Resolving EEG marker stream...")
        streams = resolve_stream('type', 'PERIODO')
        snum = 0
        if manual_stream_select:
            for i,s in enumerate(streams):
                print(i,s.name())
            snum = input("Select EEGStateAdapter stream: ")
        self.inlet = StreamInlet(streams[int(snum)])
        # launch thread
        self.eeg_thread_event.set()
        thread = Thread(target=self.eeg_rx_thread)
        thread.start()
        
    def stop_eeg_thread(self):
        self.imprintAdapter.stop_imprint_thread()
        self.rpvAdapter.stop_rpv_thread()
        self.eeg_thread_event.clear()
        
    def clear_caches(self, clear_subadapters=False):
        self.rpv_data_dict = dict()
        self.imprint_data_dict = dict()
        self.eeg_data_cache = list()
        if clear_subadapters:
            self.rpvAdapter.get_data()
            self.imprintAdapter.get_data()

    def eeg_rx_thread(self):
        '''
        Receiver will need to select correct stream, then continuously accept and 
        process commands as they arrive.
        '''
        
        rx_counter = 0
        fifo_idx = np.linspace(0,self.eeg_fifo_len-1,self.n_spectrogram_timepoints).astype(int)
        while self.eeg_thread_event.isSet():
    
            # get command
            eeg_periodo, timestamp = self.inlet.pull_sample(timeout=1)
            if eeg_periodo == None: continue #if timed out, check if thread is sitll alive
            
            assert len(eeg_periodo) == self.num_channels * self.num_freqs #lsl output is flattened periodo
            
            # add new periodogram to fifo
            self.eeg_fifo.append(eeg_periodo)
            
            # inc rx count
            rx_counter += 1
            
            # cache if apt.
            if (len(self.eeg_fifo) == self.eeg_fifo_len) and (rx_counter % self.cache_interval == 0):
                self.eeg_data_cache += [(np.asarray(self.eeg_fifo)[fifo_idx,:], timestamp)]
            
            
    def sync_data(self, _inputs, labels):
        '''
        assume inputs and labels both lists of tuples with first val value, second val timestamp.
        
        assume we have many more inputs than labels
        
        strategy is to search for the closest label for each input, also we can only,
        use one input per label so if the next closest is further than a removed one 
        then we would like to collect statistics on that.
        
        '''   
        
        ts_diffs = []    
        synced_inputs = []
        synced_outputs = []
                 
        # copy inputs
        inputs = list(_inputs)
        previously_used = list()
        
        # loop over labels to find closest input
        for label in labels:
            
            # extract timestamps to array
            inputs_ts = np.asarray([i[1] for i in inputs])
            
            # extract label ts
            label_ts = label[1]
            
            # find nearest input to label
            inputs_ts = abs( inputs_ts - label_ts )
            amin = np.argmin(inputs_ts)

            # check if the value at this index has been used before; if so, skip the label (undefined behaviour)
            if amin in previously_used: continue
            previously_used.append(amin)

            closest_input = inputs[amin][0]
            
            # collect metrics
            ts_diffs += [inputs_ts[amin]]
            
            # add pair
            #synced_pairs += [(closest_input, label[0])] 
            synced_inputs += [closest_input]
            synced_outputs += [label[0]]
            
            # delete used input
#            del inputs[amin]
            
        return {'inputs': synced_inputs, 'labels': synced_outputs, 'diffs': ts_diffs}
        
if __name__ == '__main__':
    
    eeg = EEGStateAdapter()
    eeg.launch_eeg_adapter()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    