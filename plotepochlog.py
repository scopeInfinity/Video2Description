import matplotlib.pyplot as plt
import numpy as np
import sys
fname = "vgg16_we_5.5_5.5_v1_100_logs.txt"
fname = "vgg16_we_D_DD_logs.txt"
fname = "vgg16_we_tanh_D_DD_logs.txt"
fname = "vgg16_tanh_gs5_logs.txt"

if len(sys.argv)>1:
    if len(sys.argv[1]) > 1:
    	fname=sys.argv[1]
step = 1
if len(sys.argv)>2:
        step=int(sys.argv[2])


print "Fname %s " % fname


my_data = []#np.genfromtxt(fname, delimiter=',')[1:]
ndata = []
mxcol = 0
firstRow = False
with open(fname,'r') as f:
    for row in f:
        if not firstRow:
            firstRow = True
            continue
        rr =[float(x) for x in row.split(',')]
        mxcol = max(mxcol,len(rr))
        my_data.append(rr)
#my_data = np.array(my_data)
#print my_data
#print "Yo"
for row in my_data:
    #print row
    #print mxcol
    #print len(row)
    #print type(row)

    rdata = row + [0]*(mxcol - len(row))
    #print rdata
    ndata.append(rdata)
    #print len(rdata)
ndata = np.array(ndata, dtype='float')
[ep,acc,loss,val_acc,val_loss,dis,val_dis] = [y[::step] for y in np.matrix.transpose(ndata)]


x = range(len(ep))
fig = plt.figure()
host = fig.add_subplot(111)
pacc = host.twinx()
ploss = host.twinx()
pdis = host.twinx()

_e,=host.plot(x,ep,color= plt.cm.viridis(0.95),label='Epochs')

_a,=pacc.plot(x,acc,'-.',label="Accuracy",color= plt.cm.viridis(0))
pacc.plot(x,val_acc,'-.',label="Val Accuracy",color= plt.cm.viridis(0.5))

_l,=ploss.plot(x,loss, '--', label="Loss", color = plt.cm.viridis(0))
ploss.plot(x,val_loss,'--', label="Val Loss",color= plt.cm.viridis(0.5))

_d,=pdis.plot(x,dis,'-',label="Distance",color= plt.cm.viridis(0.75))
pdis.plot(x,val_dis,'-',label="Val Distance",color= plt.cm.viridis(0.95))


ploss.legend(loc='lower right')
pacc.legend(loc='lower left')
pdis.legend(loc='upper left')


#host.spines['left'].set_position(('outward', 60))      
#pacc.spines['left'].set_position(('outward', 10))      
ploss.spines['right'].set_position(('outward', 30))      
pdis.spines['right'].set_position(('outward', 60))      



host.yaxis.label.set_color(_e.get_color())
ploss.yaxis.label.set_color(_l.get_color())
pacc.yaxis.label.set_color(_a.get_color())
pdis.yaxis.label.set_color(_d.get_color())

#plt.savefig("plot.png", bbox_inches='tight')

plt.show()
