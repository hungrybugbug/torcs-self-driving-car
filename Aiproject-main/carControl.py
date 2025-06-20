# CarControl class:
# - Holds control parameters of a car (accel, brake, gear, steer, clutch, focus, meta).
# - Provides set/get methods for each parameter.
# - Converts parameters to message format using msgParser.

import msgParser

class CarControl(object):
    '''
    An object holding all the control parameters of the car
    '''
    # TODO: Add range checks for set parameters

    def __init__(self, accel=0.0, brake=0.0, gear=1, steer=0.0, clutch=0.0, focus=0, meta=0):
        '''Constructor'''
        self.parser = msgParser.MsgParser()
        
        self.actions = {}

        self.accel = accel
        self.brake = brake
        self.gear = gear
        self.steer = steer
        self.clutch = clutch
        self.focus = focus
        self.meta = meta
    
    def toMsg(self):
        '''Convert control actions to a message format'''
        self.actions = {
            'accel': [self.accel],
            'brake': [self.brake],
            'gear': [self.gear],
            'steer': [self.steer],
            'clutch': [self.clutch],
            'focus': [self.focus],
            'meta': [self.meta],
        }
        
        return self.parser.stringify(self.actions)
    
    def setAccel(self, accel):
        self.accel = accel
    
    def getAccel(self):
        return self.accel
    
    def setBrake(self, brake):
        self.brake = brake
    
    def getBrake(self):
        return self.brake
    
    def setGear(self, gear):
        self.gear = gear
    
    def getGear(self):
        return self.gear
    
    def setSteer(self, steer):
        self.steer = steer
    
    def getSteer(self):
        return self.steer
    
    def setClutch(self, clutch):
        self.clutch = clutch
    
    def getClutch(self):
        return self.clutch
    
    def setMeta(self, meta):
        self.meta = meta
    
    def getMeta(self):
        return self.meta
