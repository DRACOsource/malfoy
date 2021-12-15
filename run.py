# -*- coding: utf-8 -*-
from v2x.config.config import (PSTATS_FILE,LOGS_DIR,
                               RESOURCE_SITE_PARAMS as rsp)
from v2x.utils.common_utils import openFiles,closeFiles
from v2x.solutions.v2x import V2XModel
from v2x.utils.graphic_utils import Graphics
import cProfile
import pstats
import os,sys,re
import numpy as np
import warnings
warnings.filterwarnings("ignore")
    
def runProfiler():
    cProfile.run('runStandAloneModel()', PSTATS_FILE)
    p = pstats.Stats(PSTATS_FILE)
    p.strip_dirs().sort_stats('cumulative').print_stats(30)
        
def runStandAloneModel(nrSites=2,train=True,evaluation=False,
                       loadModel=False,steps=8000,rebid=True,curious=False,
                       interval=None,cumstep=0,endOfGame=5000,
                       sumoDataOffset=0,ca=True):
    nrRsu = 1
    
    if interval is not None:
        filedict = openFiles(additional=[interval,nrSites,str(train)],
                             trainable=(train and not evaluation))
    else:
        filedict = openFiles(additional=[nrSites,str(train)],
                             trainable=(train and not evaluation))
    mdl = V2XModel(filedict,nrSites=nrSites,nrRsu=nrRsu,train=train,
                   evaluation=evaluation,loadModel=loadModel,rebid=rebid,
                   curious=curious,extRewardInterval=interval,
                   cumstep=cumstep,endOfGame=endOfGame,
                   sumoDataOffset=sumoDataOffset,ca=ca)
    
    iterations = len(rsp.resourceCapa)
    chpt = list(np.arange(start=steps/iterations,stop=steps,
                     step=steps/iterations,dtype=int))[0:iterations-1]
    for i in range(steps):
        mdl.step()
        if i in chpt:
            actor_learning_rate = []
            for v in mdl.vehicles.values():
                actor_learning_rate.append(v.priceMdl.actor_learning_rate)
            print('actor learning rate: {},{}.'.format(
                 min(actor_learning_rate),max(actor_learning_rate)))
    closeFiles(filedict)
    del mdl.trace.context
    return mdl

#%%
if __name__ == '__main__':

    train = True
    nrSites = 2
    evaluation = False
    loadModel = True
    steps = 200000
    sumoDataOffset = 50000
    path = LOGS_DIR
    curious = True
    rebid = True
    interval = 1
    outputThres = 1000
    repetition = 1
    ca = True # if true: run attention model (credit assignment)

    if sys.argv[1]=='draw':
        path = LOGS_DIR
        try:
            path = os.path.join(path,sys.argv[2])
        except:
            pass
        graph = Graphics(path)
        res = graph.drawPerformance(drawPerformanceOnly=True,
                      target='totalSuccessRatio',textOutput=True,
                      outputThres=outputThres,endOfGame=steps,
                      legends='best',stepRange=(2000,200000),
                      decimals=2,yaxisTick='right',orderHue='by algorithm',
                      density=1,ci='sd')
    else:
        if 'rial' in sys.argv[1]:
            train = False
        if 'eval' in sys.argv[1]:
            evaluation = True
            if 'sameInput' in sys.argv[1]:
                sumoDataOffset += steps
        if 'rebid' not in sys.argv[1]:
            rebid = False
        if 'noAttention' in sys.argv[1]:
            ca = False
        try:
            if 'interval' in sys.argv[1]:
                location = re.search('=',sys.argv[1]).span()[1]
                try:
                    end = location + re.search('_',sys.argv[1][location:]).span()[1]
                    interval = int(sys.argv[1][location:end-1])
                except:
                    interval = int(sys.argv[1][location:])
        except:
            pass
            
        try:
            path = os.path.join(path,sys.argv[3])
        except:
            pass

        if interval==1:
            curious = False

        for i in range(repetition):
            runStandAloneModel(nrSites=nrSites,
                    train=train,evaluation=evaluation,
                    loadModel=loadModel,steps=steps,rebid=rebid,
                    curious=curious,interval=interval,cumstep=i*steps,
                    endOfGame=steps,sumoDataOffset=sumoDataOffset,ca=ca)