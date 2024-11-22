import numpy as np
import Fluid2d as Fd

class NumericalFlux:
    Uin : Fd.Fluid_2d
    Uout : Fd.Fluid_2d
    n : np.array
    DIM = 2

    def __init__(self,Uin,Uout,n):
        self.Uin = Uin
        self.Uout = Uout
        self.n = n
    '''
    Local LF flux Dim=2 
    '''
    def LLF(self)->np.array:
        max_speedin = max(abs(np.dot(self.Uin.u,self.n)-self.Uin.sound_speed()),abs(np.dot(self.Uin.u,self.n)+self.Uin.sound_speed()))
        max_speedout = max(abs(np.dot(self.Uout.u,self.n)-self.Uout.sound_speed()),abs(np.dot(self.Uout.u,self.n)+self.Uout.sound_speed()))
        max_speed = max(max_speedin,max_speedout)
        #max_speed=max(np.linalg.norm(self.Uout.u)+self.Uout.sound_speed(),np.linalg.norm(self.Uin.u)+self.Uin.sound_speed())
        Fu = np.zeros(self.DIM+2,dtype=float)
        Fu[:] += (self.Uin.flux()+self.Uout.flux())@self.n
        Fu[:] -= max_speed*(self.Uout.U-self.Uin.U)
        return 0.5*Fu
