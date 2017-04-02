from recordstore import Recorder
import time

r = Recorder(2)
r.record()
time.sleep(2)
r.play() 
