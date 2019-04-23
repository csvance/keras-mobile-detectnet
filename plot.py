from model import MobileDetectnetModel
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model = MobileDetectnetModel.create(weights=None)
model.plot()