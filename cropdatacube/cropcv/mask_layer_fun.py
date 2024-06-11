import numpy as np
import math
from ..utils.distances import euclidean_distance

def get_boundingboxfromseg(mask):
    
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    
    return([xmin, ymin, xmax, ymax])

def getmidleheightcoordinates(pinit,pfinal,alpha):

  xhalf=math.sin(alpha) * euclidean_distance(pinit,pfinal)/2 + pinit[0]
  yhalf=math.cos(alpha) * euclidean_distance(pinit,pfinal)/2 + pinit[1]
  return int(xhalf),int(yhalf)

def getmidlewidthcoordinates(pinit,pfinal,alpha):

  xhalf=pfinal[0] - math.cos(alpha) * euclidean_distance(pinit,pfinal)/2
  yhalf=pinit[1] - math.sin(alpha) * euclidean_distance(pinit,pfinal)/2
  return int(xhalf),int(yhalf)

