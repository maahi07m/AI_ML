import numpy as np
from matplotlib import pyplot as plt
data = np.array([300,3,'BOM',20.6,100,4,'DEL',30.4,150,5,'CHN',32.5,320,3,'BLR',82.4,580,3,'BOM',92.5,280,4,'CHN',32.6,380,2,'BLR',23.6,264,3,'BLR',21.8,253,3,'DEL',25.6])
data = data.reshape(9,4)
#print(data)
x = data[:,[0,1,2]]
y = data[:,[3]]
def train_model(x,y):
    theta = np.zeros((x,shape[1],1))
    theta = optimize_weights_using_gradient_descent(x,y,theta,10,0.0001)

