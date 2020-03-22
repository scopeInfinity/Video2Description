from enum import Enum

class ModelWeightsStatus(Enum):
    NO_INFO         = 0
    SUCCESS         = 1
    MODEL_NOT_FOUND = 2
    WIP             = 3