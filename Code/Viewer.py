import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mayaView(num, path = "./data"):
    """
    num is a str eg. "20-1", "400"
    loads images from ./data
    """
    assert type(num) == str
    x = np.load(path+'/xdata{}.dat'.format(num))
    y = np.load(path+'/ydata{}.dat'.format(num))
    z = np.load(path+'/zdata{}.dat'.format(num))
    density = np.load(path+'/density{}.dat'.format(num))

    figure = mlab.figure('DensityPlot')
    mag = density/np.amax(density)

    n = num[0]
    l = num[1]
    if num[2] == "-":
        m = num[2:]
    else:
        m = num[2]

    pts = mlab.points3d(mag, opacity=0.5, transparent=True)
   # pts = mlab.contour3d(mag,opacity =0.5)
    #mlab.title('n= {}, l = {}, m= {}'.format(n,l,m), size= 20)
    mlab.text(0.5,0.2, 'n= {}, l = {}, m= {}'.format(n,l,m), opacity = 1)

    mlab.colorbar(orientation = 'vertical')
    mlab.axes()
    mlab.savefig('/home/ada/Documents/SUTD/Term 3/Physical Chemistry/Schrodinger-Assignment/Images/'+num+'.png')
    #mlab.show()

def matplotView(num):
    assert type(num) == str
    x = np.load('./data/xdata{}.dat'.format(num))
    y = np.load('./data/ydata{}.dat'.format(num))
    z = np.load('./data/zdata{}.dat'.format(num))
    mag = np.load('./data/density{}.dat'.format(num))
    fig=plt.figure()
    fig.suptitle(num,fontsize=14)
    ax=fig.add_subplot(111,projection='3d')
    for a in range(0,len(mag)):
        for b in range(0,len(mag)):
            for c in range(0,len(mag)):
                ax.scatter(x[a][b][c],y[a][b][c],z[a][b][c],marker='o',alpha=(mag[a][b][c]/np.amax(mag)))
    plt.savefig('/home/ada/Documents/SUTD/Term 3/Physical Chemistry/Schrodinger-Assignment/Images/'+num+'.png')
    #plt.show()


mayaView("320")
'''
if __name__ == "__main__":
    import time
    for n in range(2,3):
        for l in range(-(n-1),n):
            for m in range(-l,l+1):
                nlm = str(n)+str(l)+str(m)
                print nlm
                try:
                    mayaView(nlm)
                    time.sleep(2)

                except:
                    print "++++++++++++++++++++++"
                    print "did not save: " + nlm
                    print "++++++++++++++++++++++"
                    continue

'''
