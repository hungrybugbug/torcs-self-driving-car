
"""
FILE: msgParser.py
This file contains the implementation of the `MsgParser` class, which is responsible for:
- Parsing received UDP messages into a dictionary format.
- Building UDP messages from a dictionary.
"""

class MsgParser(object):
    '''
    A parser for received UDP messages and building UDP messages
    '''
    def __init__(self):
        '''Constructor'''
        pass  # Added 'pass' to prevent empty function error
    
    def parse(self, str_sensors):
        '''Return a dictionary with tags and values from the UDP message'''
        sensors = {}
        
        b_open = str_sensors.find('(')
        
        while b_open >= 0:
            b_close = str_sensors.find(')', b_open)
            if b_close >= 0:
                substr = str_sensors[b_open + 1: b_close]
                items = substr.split()
                if len(items) < 2:
                    print("Problem parsing substring:", substr)  # Fixed print
                else:
                    sensors[items[0]] = items[1:]  # More Pythonic
                b_open = str_sensors.find('(', b_close)
            else:
                print("Problem parsing sensor string:", str_sensors)  # Fixed print
                return None
        
        return sensors
    
    def stringify(self, dictionary):
        '''Build a UDP message from a dictionary'''
        msg = ''
        
        for key, value in dictionary.items():
            if value is not None and value[0] is not None:  # Fixed None check
                msg += '(' + key + ' ' + ' '.join(map(str, value)) + ')'
        
        return msg
