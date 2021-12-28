from enum import Enum

class Hyperparams(Enum):
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 10
    NUM_CLASSES = 10

class Constants(Enum):
    MODEL_PATH = "model.pth"
    WHITE = "#FFF"
    BLACK = "#000"
    HEIGHT = 400
    WIDTH = 400
    
