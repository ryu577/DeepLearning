import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageMath

## First, generate some test data.

def get_data(sam=100):
    r = 0.1
    theta1 = 0
    theta2 = np.pi/4
    theta3 = np.pi/2
    theta4 = 3*np.pi/4
    X = []
    y = []
    Y = []
    for i in range(sam):
        x1=np.array([r*np.cos(theta1), r*np.sin(theta1)])+1000+np.random.normal(size=2)*10
        x2=np.array([r*np.cos(theta2), r*np.sin(theta2)])+1000+np.random.normal(size=2)*10
        x3=np.array([r*np.cos(theta3), r*np.sin(theta3)])+1000+np.random.normal(size=2)*10
        x4=np.array([r*np.cos(theta4), r*np.sin(theta4)])+1000+np.random.normal(size=2)*10
        X.append(x1)
        X.append(x2)
        X.append(x3)
        X.append(x4)
        y.append(0)
        y.append(1)
        y.append(2)
        y.append(3)
        Y.append(np.array([1,0,0,0]))
        Y.append(np.array([0,1,0,0]))
        Y.append(np.array([0,0,1,0]))
        Y.append(np.array([0,0,0,1]))
        #theta1 += np.pi/120
        #theta2 += np.pi/120
        #theta3 += np.pi/120
        #theta4 += np.pi/120
        r += 3
    return [np.array(X), np.array(y), np.array(Y)]


rgbs = [(255,255,255),(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
basedir = 'C:\\Users\\ropandey\\Documents\\GitHub\\DeepLearning\\'
im = Image.new("RGB", (2048, 2048), (1,1,1))
draw = ImageDraw.Draw(im,'RGBA')

[X,y,Y] = get_data(1000)

for i in range(len(y)):
    draw.ellipse((X[i][0]-3,X[i][1]-3,X[i][0]+3,X[i][1]+3), fill = rgbs[y[i]])
im.save(basedir + "im" + str(1) + ".png")



