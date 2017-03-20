import Week5 as w
import numpy as np
import os

def dump():
    print "dump start!"
    N = 20
    roa = 10
    if not os.path.exists('./data'):
        os.makedirs('./data')
    for n in range(2,5):
        print "for n: " + str(n)
        for l in range(0,n):
            print "\tfor l: " + str(l)
            if l == 0:
                print "\t\tfor m: " + str(0)
                x,y,z, mag =w.hydrogen_wave_func(n,l,0,roa,N,N,N)
                x.dump("./data/xdata{}{}{}.dat".format(n,l,0,roa,N,N,N))
                y.dump("./data/ydata{}{}{}.dat".format(n,l,0,roa,N,N,N))
                z.dump("./data/zdata{}{}{}.dat".format(n,l,0,roa,N,N,N))
                mag.dump("./data/density{}{}{}.dat".format(n,l,0,roa,N,N,N))

            elif l > 0:
                for m in range(-l,l+1):
                    try:
                        print "\t\tfor m: " + str(m)
                        x,y,z, mag =w.hydrogen_wave_func(n,l,m,roa,N,N,N)
                        x.dump("./data/xdata{}{}{}.dat".format(n,l,m,roa,N,N,N))
                        y.dump("./data/ydata{}{}{}.dat".format(n,l,m,roa,N,N,N))
                        z.dump("./data/zdata{}{}{}.dat".format(n,l,m,roa,N,N,N))
                        mag.dump("./data/density{}{}{}.dat".format(n,l,m,roa,N,N,N))
                    except:
                        continue

if __name__ == "__main__":
    dump()
