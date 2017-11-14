import matplotlib.pyplot as plt
import numpy as np
import sys
fname = "vgg16_we_5.5_5.5_v1_100_logs.txt"
if len(sys.argv)>1:
    fname=sys.argv[1]
print "Fname %s " % fname
my_data = np.genfromtxt(fname, delimiter=',')[1:]
[ep,acc,loss,val_acc,val_loss] = np.matrix.transpose(my_data)

x = range(len(ep))
fig = plt.figure()
host = fig.add_subplot(111)
pacc = host.twinx()
ploss = host.twinx()

_e,=host.plot(x,ep,color= plt.cm.viridis(0.95),label='Epochs')
_a,=pacc.plot(x,acc,'-',label="Accuracy",color= plt.cm.viridis(0))
_l,=ploss.plot(x,loss, '--', label="Loss", color = plt.cm.viridis(0))
pacc.plot(x,val_acc,'-',label="Val Accuracy",color= plt.cm.viridis(0.5))
ploss.plot(x,val_loss,'--', label="Val Loss",color= plt.cm.viridis(0.4))
ploss.legend(loc='lower right')
pacc.legend(loc='lower left')

#host.spines['left'].set_position(('outward', 60))      
#pacc.spines['left'].set_position(('outward', 10))      
ploss.spines['right'].set_position(('outward', 50))      



host.yaxis.label.set_color(_e.get_color())
ploss.yaxis.label.set_color(_l.get_color())
pacc.yaxis.label.set_color(_a.get_color())

#plt.savefig("pyplot_multiple_y-axis.png", bbox_inches='tight')

plt.show()
