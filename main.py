from recordstore import Recorder
import time

r = Recorder(2)
# r.train_svm("FinalDB")
# r.record()
# time.sleep(2)
# r.play()
r.detect_emotion() 
# r.extract_features_from_input_voice()
