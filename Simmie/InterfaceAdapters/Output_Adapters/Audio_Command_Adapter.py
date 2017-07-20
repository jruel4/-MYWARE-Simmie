'''
Destination: AudioSlave

Description: Audio commands for entrainment oscillators from PI network.

Consumers: AudioSlave
'''

'''
AudioSlave CommandAdapter rx spec:
    
    osc_id = self.OscVals[command_set[self.OscIdx] - self.OscOffset]
    vol = self.DynamicsVals[command_set[self.DynamicsIdx] - self.DynamicsOffset]
    base = self.FMVals[command_set[self.FMIdx] - self.FMOffset]
    bleat = self.AMVals[command_set[self.AMIdx] - self.AMOffset]
    
    Initial command logic will suppose no coupled oscillators and 4 uncoupled.
    We need to map each command to AudioSlave command, which simply means 
    that we need an interface method to feed the command set into.
    
    - Dynamics:
        - 0 = 0.5
        - 1 = 0.3
        - 2 = 0.1
        
    - FM:
        - 3 = get_n_key_freq(35)
        - 4 = get_n_key_freq(36)
        .
        .
        .
        - 32 = get_n_key_freq(64)
        
    - AM:
        - 33 = 0 Hz
        - 34 = 1 Hz
        .
        .
        .
        - 62 = 29 Hz
        
    - Oscillator (Uncoupled):
        - 63 = Oscillator_1
        - 64 = Oscillator_2
        - 65 = Oscillator_3
        - 66 = Oscillator_4
        
    # TODO
    - Oscillator (Coupled):
        - 67 = IAF (AM locked to IAF)
        - 68 = BBF (Dynamics locked to Beta)
        - 69 = ABF (Dynamics locked to IAF)
        - 70 = TBF (Dynamics locked to Theta)
    '''