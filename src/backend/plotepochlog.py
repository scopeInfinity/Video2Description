import matplotlib.pyplot as plt
import numpy as np
import sys

MXPOINT = 100
assert len(sys.argv)>=2
fname=sys.argv[1]
showtrain = True
if  len(sys.argv)>=3:
    showtrain = (sys.argv[2] == 'yo')
showval = True
if  len(sys.argv)>=4:
    showval = (sys.argv[3] == 'yo')
showepoch = True
if len(sys.argv)>=5:
    showepoch = (sys.argv[4] == 'yo')

print("Fname %s " % fname)

batch = []
loss = []
acc  = []
val_loss = []
val_acc  = []

ndata = []
with open(fname,'r') as f:
    for row in f:
        rr =[float(x) for x in row.split(',')]
        ndata.append(rr)

ndata = np.array(ndata, dtype='float')
print(np.shape(ndata))
step = 1
if len(ndata[0]) > MXPOINT:
    step = len(ndata[0]) // MXPOINT
[batch, loss, acc, val_loss, val_acc,cider,bleu4,rouge,meteor] = [y[::step] for y in np.matrix.transpose(ndata)][:9]

x = range(len(batch))
fig = plt.figure()
host = fig.add_subplot(111)
pscores = host.twinx()
pacc = host.twinx()
ploss = host.twinx()

if showepoch:
    _b,=host.plot(x,batch,color= plt.cm.viridis(0.95),label='Batches')

if showtrain:
    _a,=pacc.plot(x,acc,'-.',label="Accuracy",color= plt.cm.viridis(0))
    _l,=ploss.plot(x,loss, '-', label="Loss", color = plt.cm.viridis(0))
if showval:
    ploss.plot(x,val_loss,'-', label="Val Loss",color= plt.cm.viridis(0.5))
    pacc.plot(x,val_acc,'-.',label="Val Accuracy",color= plt.cm.viridis(0.5))
if showtrain or showval:
    ploss.legend(loc='lower right')
    pacc.legend(loc='lower left')
    ploss.spines['right'].set_position(('outward', 30))      

score_total = cider+bleu4+rouge+meteor
pscores.plot(x,cider,'-', label="Cider",color= plt.cm.viridis(0.0))
pscores.plot(x,bleu4,'-', label="Bleu4",color= plt.cm.viridis(0.2))
pscores.plot(x,rouge,'-', label="Rouge",color= plt.cm.viridis(0.4))
pscores.plot(x,meteor,'-', label="Meteor",color= plt.cm.viridis(0.6))
pscores.plot(x,score_total,'-', label="Total",color= plt.cm.viridis(0.8))
pscores.legend(loc='upper left')


#host.yaxis.label.set_color(_b.get_color())
#ploss.yaxis.label.set_color(_l.get_color())
#pacc.yaxis.label.set_color(_a.get_color())

#plt.savefig("plot.png", bbox_inches='tight')

best_iter = np.argmax(score_total)
print("Best Iteration %d " % best_iter)
print("\tCIDER  %.4f " % cider[best_iter])
print("\tBLEU4  %.4f " % bleu4[best_iter])
print("\tROUGE  %.4f " % rouge[best_iter])
print("\tMETEOR %.4f " % meteor[best_iter])
print("\tTotalScore %.4f " % score_total[best_iter])



plt.show()
