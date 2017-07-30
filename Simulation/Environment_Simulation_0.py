#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:49:41 2017

@author: marzipan
"""

'''
Simulate minimal "brain dynamics". E.g. simply map input action to a frequency
and output a fake eeg signal at that same frequency.
'''

# We need a single thread that acquires action commands and outputs fake eeg data.

# Imports
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream
from threading import Thread, Event
from LSLUtils import Periodogram as Fake_Periodogram

# Utility functions
def get_n_key_freq(n):
    return 2.0**((n-49)/12.0) * 440.0

# TODO write a proper test with a melody and other predictable elements

class EnvironmentSimulator:
    '''
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
    DynamicsCommands = list(range(0,3))
    FMCommands = list(range(3,33))
    AMCommands = list(range(33,63))
    OscillatorUncoupledCommands = list(range(63,67))
    OscillatorCoupledCommands = list(range(67,71))
    
    def __init__(self):
        
        self.osc_id = 1
        self.overrid_gui = False
        
        # Constants
        self.DynamicsIdx = 0
        self.FMIdx = 1
        self.AMIdx = 2
        self.OscIdx = 3
        
        self.DynamicsOffset = 0
        self.FMOffset = 3
        self.AMOffset = 33
        self.OscOffset = 63
        
        self.DynamicsVals = [0.0, 0.01, 0.05] # TODO Relative vol (iso_vol)
        self.FMVals = [get_n_key_freq(35+i) for i in range(30)] # Hz
        self.AMVals = [i for i in range(30)] # Hz
        self.OscVals = [i for i in range(1,5)] # Id
        
        self.CmdMapKeys = ['DYNAMICS_CMD', 'AM_CMD', 'FM_CMD']
        
        self.CmdValMap = {'DYNAMICS_CMD': self.DynamicsVals,
                          'AM_CMD': self.AMVals,
                          'FM_CMD': self.FMVals}
        
        self.CmdOffsetMap = {'DYNAMICS_CMD': self.DynamicsOffset,
                             'AM_CMD': self.AMOffset,
                             'FM_CMD': self.FMOffset}
        
        # Vars
        self.command_thread_event = Event()
        
    def execute_single_command(self, command):
        '''
        Accept audio source (asrc) object from AudioSlave and executes a new command
        set on it.
        '''

        
        # Map single command index to type
        if command in EnvironmentSimulator.DynamicsCommands:
            vol = self.DynamicsVals[command - self.DynamicsOffset]
            #TODO use the vol to influence output power...
 
        elif command in EnvironmentSimulator.AMCommands:
            bleat = self.AMVals[command - self.AMOffset]
            
            #TODO set the fake eeg output to this bleat frequency
            self.set_fake_eeg_properties(bleat)

        elif command in EnvironmentSimulator.FMCommands:
            base = self.FMVals[command- self.FMOffset]
            #TODO construct hypothesis regarding effect of base frequency on entrainment.

        elif command in EnvironmentSimulator.OscillatorCoupledCommands:
            pass
            
        elif command in EnvironmentSimulator.OscillatorUncoupledCommands:
            self.osc_id = self.OscVals[command - self.OscOffset]
            
    def simulation_thread(self):
        '''
        Receiver will need to select correct stream, then continuously accept and 
        process commands as they arrive.
        '''
        
        self.rx_count = 0
        while self.command_thread_event.isSet():
    
            # get command
            command_set, timestamps = self.inlet.pull_sample()
            
            if self.rx_count % 5 == 0:
                print("rx cmd: ",command_set)
            self.rx_count += 1
            
            # execute command
            if len(command_set) == 4:
                self.execute_command_set(command_set)     
            elif len(command_set) == 1:
                self.execute_single_command(command_set[0])
            else:
                print("Invalid command set size: ",command_set)
                
        print("Ending command rx loop")

    def launch_simulator(self, manual_stream_select=True):
        
        # Select the command thread to receive inputs from
        streams = resolve_stream('type', 'AudioCommands')
        snum = 0
        if manual_stream_select:
            for i,s in enumerate(streams):
                print(i,s.name())
            snum = input("Select desired stream: ")
        self.inlet = StreamInlet(streams[snum])
        self.command_thread_event.set()
        thread = Thread(target=self.simulation_thread)
        thread.start()
        
        # Launch the fake data simulator to send outputs to
        #TODO use LSL Utils
        self.output_data_simulator = Fake_Periodogram()
        self.output_data_simulator._launch_thread()
        self.output_data_simulator.set_peaks_amps([1.0], [.2])
        
    def set_fake_eeg_properties(self, freq):
        '''
        Set dominant eeg frequency to use in eeg data synthesis
        '''
        #TODO set fake freqs
        self.output_data_simulator.set_peaks_amps([freq], [0.2])
        
if __name__ == "__main__":
    #TODO write test 
    pass        
        
      
        
        
        
        
        
        
        
        
        
        
        
        
        