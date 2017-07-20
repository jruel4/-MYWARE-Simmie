'''
Source: AudioSlave via DataManager

Description: Labeled training data generated by proctor for PI network.

subprocesses: P2 input
'''

'''
Imprinting should be executed in batches, not real-time. So we need to:
    
    - Collect the training data as it is generated.
    
    - Decide when it's ready to batch train and feed it to Simmie's PI net.
    
Data collection:
    
    - This data will come from the proctor via the AudioSlave's Imprint Command
      adapter. 
      
    - After N samples are collected, feed / make available to Simmie.
    
    - #TODO need to associate timestamped data with EEG
      
'''

from threading import Thread, Event
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream

class ImprintAdapter:
    
    def __init__(self):
        self.command_thread_event = Event()
        self.imprint_data_cache = list()
        
    def retrieve_training_batch(self):
        #TODO what numpy format is good?
        pass
    
    def launch_command_adapter(self, manual_stream_select=True): 
        streams = resolve_stream('type', 'AudioCommands')
        snum = 0
        if manual_stream_select:
            for i,s in enumerate(streams):
                print(i,s.name())
            snum = input("Select desired stream: ")
        self.inlet = StreamInlet(streams[snum])
        # launch thread
        self.command_thread_event.set()
        thread = Thread(target=self.command_rx_thread)
        thread.start()
        
    def stop_imprint_thread(self):
        self.command_thread_event.clear()

    def command_rx_thread(self):
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
        
        
        
        
        
        
        
        
        
        
        
        