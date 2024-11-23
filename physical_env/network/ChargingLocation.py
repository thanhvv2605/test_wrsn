import numpy as np
from sklearn.cluster import KMeans
import math

from sklearn.metrics import silhouette_score

class ChargingLocation:
    def __init__(self, id, centroid):
        
        self.id = id
        self.charging_location = centroid
