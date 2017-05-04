from audioFeatureExtraction import FeatureExtractor
import time

r = FeatureExtractor(2)
# r.read_data("../FinalDB")
r.train_svm()
# r.record()
# time.sleep(2)
# r.play()
# r.find_mfcc("test.wav")
# r.find_energy_features("test.wav")
# r.detect_emotion("FinalDB/Testing Set") 