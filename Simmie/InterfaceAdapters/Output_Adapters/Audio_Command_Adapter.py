'''
Destination: AudioSlave

Description: Audio commands for entrainment oscillators from PI network.

Consumers: AudioSlave
'''
'''
Input: 
    
    - One-hot index from policy network
    
Output: 
    
    - audio command set over lsl as list = [dynamics, fm, am, osc]
    
Processing:
    
    - The output from the one hot vector will already be "encoded" so we can just
      send it over LSL. But we do need to check to make sure that correct order
      of commands we enforced.
          
'''

# Imports
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream

class AudioCommandAdapter:
    
    # Static vars - these should actually be shard with AudioSlave Adapter
    DynamicsCommands = list(range(0,3))
    FMCommands = list(range(3,33))
    AMCommands = list(range(33,63))
    OscillatorUncoupledCommands = list(range(63,67))
    
    def __init__(self):        
        self.command_set = list()
        self.info = StreamInfo('MartianWearables', 'AudioCommands', 4, 4, 'int32', 'AudioCommandUID002')
        self.outlet = StreamOutlet(self.info)
        
    def get_valid_audio_commands(self):
        return  AudioCommandAdapter.DynamicsCommands +\
                AudioCommandAdapter.FMCommands +\
                AudioCommandAdapter.AMCommands +\
                AudioCommandAdapter.OscillatorUncoupledCommands

    def submit_command(self, command_idx):
        '''
        To be called by policy client after outputing a command each time.
        '''
        
        #TODO collect commands into local buf, every 4, check them and send.
        self.command_set += [command_idx]
        if len(self.command_set) == 4:
            if not self.check_valid_set():
                print("ERROR: Command adapter received invalid set")
                self.command_set = list()
                return False
            
            # send over network
            self.outlet.push_sample(self.command_set)
            
            # Clear
            self.command_set = list()
            
        return True
            
    def check_valid_set(self):
        dynamic_first = self.command_set[0] in AudioCommandAdapter.DynamicsCommands
        fm_second = self.command_set[1] in AudioCommandAdapter.FMCommands
        am_third = self.command_set[2] in AudioCommandAdapter.AMCommands
        osc_fourth = self.command_set[3] in AudioCommandAdapter.OscillatorUncoupledCommands
        
        return dynamic_first and fm_second and am_third and osc_fourth
    
if __name__ == "__main__":
    
    adapter = AudioCommandAdapter()
    
    # test good data
    good = [1,30,34,65]
    for v in good:
        print(adapter.submit_command(v))
    
    bad = [17,30,2,65]
    for v in bad:
        print(adapter.submit_command(v))
    
        


        
        
        
        
        
        
        
        
        

    














