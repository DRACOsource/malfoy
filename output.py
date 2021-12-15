# -*- coding: utf-8 -*-
from v2x.config.config import (LOGS_DIR, PRICE_MODEL_PARAMS as pmp,
                               CURIOUS_MODEL_PARAMS as cmp)
from v2x.utils.graphic_utils import Graphics
from v2x.utils.name_utils import ColumnHead as ch
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from scipy.stats import t as tDistr
import re,sys,os
import warnings
warnings.filterwarnings("ignore")

#LEARNED = c.LEARNED
#RANDOM = c.RANDOM

def _changeColumnName(data,oldName,newName):
    cols = list(data.columns)
    if oldName in cols:
        cols[cols.index(oldName)] = newName
        data.columns = cols
    return data

def outputBackOffCharts(subdir=['25-25-60-rebid=1'],minStaging=1,
                        outputGraph=True,path=LOGS_DIR,size=(5,4)):
    name='finishedBids'
    target = ch.BIDPRICE
    folder = subdir[0]
    capa = int(folder[0:re.search('-',folder).span()[0]]) * 2
    rebid = folder[-7:]
    data = pd.DataFrame()
    for filepath in subdir:
        path = os.path.join(path,filepath)
        graph = Graphics(path)
        data0 = graph._collectData(name)
        if len(data)==0:
            data = data0
        else:
            data = pd.concat([data,data0],axis=0)
     
    if len(data)==0:
        return None
    
    col = [ch.STEP,ch.TOTALACTIVEBIDS,ch.SORTIDX,ch.BATCHCREATE,ch.CREATETIME]
    for c in col:
        try:
            data[c] = data[c].astype(int)
        except:
            continue
    col = [target,ch.BIDPAYMENT,ch.CARBUDGET,ch.BIDVALUE]
    for c in col:
        try:
            data[c] = data[c].astype(float)
        except: 
            continue
    
    nrSites = '2sites'
    df = data.loc[data[ch.NRSITES]==nrSites]
    df[ch.VEHICLE] = df[ch.BIDID].apply(lambda x: 
                                    x[0:re.search('_',x).span()[0]])
    maxStep = max(df[ch.STEP])
    df1 = df[df[ch.STEP]>maxStep-5000]
        
    # average staging time
    df1[ch.BACKOFFTIME] = df1.apply(lambda row: 
                        row[ch.CREATETIME]-row[ch.BATCHCREATE],axis=1)
    df1 = df1[df1[ch.BACKOFFTIME]> minStaging] 
    
    try:
        df1[ch.ADMITTED] = np.where(df1[ch.DECISION]=='admitted',
           'admitted','failed')
    except: 
        df1[ch.ADMITTED] = np.where(df1[ch.SUCCESS]==1,'admitted','failed')
    ttl = df1.groupby([ch.VEHICLE,ch.BIDQOS,ch.CARBUDGET],as_index=False)\
        .agg({ch.BIDID:'size'}).rename(columns={ch.BIDID:ch.TOTALBYQOS})
    df1 = pd.merge(df1,ttl,on=[ch.VEHICLE,ch.BIDQOS,ch.CARBUDGET],copy=False)
    
    df2 = df1[[ch.VEHICLE,ch.TRAINED,ch.BIDQOS,ch.BIDID,ch.BIDPRICE,
               ch.BACKOFFTIME,ch.TOTALBYQOS,ch.CARBUDGET]].groupby(
              [ch.VEHICLE,ch.TRAINED,ch.BIDQOS,ch.CARBUDGET],
              as_index=False).agg({ch.BIDID:'size',ch.BIDPRICE:'mean',
              ch.BACKOFFTIME:'mean',ch.TOTALBYQOS:'mean'})
    df2 = _changeColumnName(df2,ch.BIDQOS,ch.DEADLINE)
    df2[ch.DEADLINE] = np.where(df2[ch.DEADLINE]=='low quality',
                                   'long deadline','short deadline')
    df2 = _changeColumnName(df2,ch.CARBUDGET,ch.BUDGET)
    df2[ch.BUDGET] = np.where(df2[ch.BUDGET]==min(df2[ch.BUDGET]),
                                   'low budget','high budget')
    df2[ch.CAPACITY] = capa
    df2.sort_values(by=[ch.DEADLINE,ch.BUDGET,ch.TRAINED,ch.BIDPRICE],
                    ascending=[True,True,True,True],inplace=True)

    df2_1 = df2[df2[ch.TRAINED]==ch.LEARNED]
    if outputGraph:
        graph._drawBoxplot(df=df2_1,x=ch.DEADLINE,y=ch.BACKOFFTIME,
            title='backoff_'+'capa='+str(capa)+'_'+rebid+'_budget',
            hue=ch.BUDGET,legends=1,ylabel=None,
            legendFontsize=8,figsize=size,
            myPalette = {'low budget':'C1','high budget':'C0'})

    df2_1[ch.PRICERANGE] = np.where(
            df2_1[ch.BIDPRICE]>np.mean(df2_1[ch.BIDPRICE]),
            'high price','low price')
    if outputGraph:    
        graph._drawBoxplot(df=df2_1,x=ch.DEADLINE,y=ch.BACKOFFTIME,
            title='backoff_'+'capa='+str(capa)+'_'+rebid+'_priceRange',
            hue=ch.PRICERANGE,legends=None,ylabel=None,
            legendFontsize=12,figsize=size,
            myPalette = {'low price':'C1','high price':'C0'})
        
    return df2_1

def getServiceLevelData(filename='servicelevel_rebid_1.csv',
                        stepRange=(300,7999),decimals=2,path=LOGS_DIR):
    filepath = os.path.join(path,filename)
#    newname = ch.MP
    try:
        servicedata = pd.read_csv(filepath,sep=';')
#        servicedata = _changeColumnName(servicedata,oldname,newname)
#        servicedata[ch.TYPE] = servicedata.apply(lambda row:
#                row[ch.ALGORITHMTYPE]+','+newname+'='+str(row[ch.MP]),axis=1)
        return servicedata
    except:
        pass
    folders = []
    rebids = ['1','5']
    for i in np.arange(25,120,step=5):
        for rebid in rebids:
            folders.append(
                    '-'.join([str(int(i)),str(int(i)),'60','rebid='])+rebid)

    servicedata = pd.DataFrame()
    for folder in folders:
        path0 = os.path.join(path,folder)
        graph = Graphics(path0)
        idx = ['_'] + [str(int(x)) for x in list(range(1,10))]
        for j in idx:
            name = 'performance' + j
            try:
                df0 = graph._getPerformanceData(name=name,stepRange=stepRange,
                                sites=['2sites'])
            except:
                print(folder)
                print(j)
            if df0 is None:
                continue
            values = graph._outputFailureComparison(data=df0,textOutput=False,
                            graphicOutput=False,stepRange=stepRange,legends=2)
            df = pd.DataFrame(columns=[ch.CAPACITY,ch.STANDARD,ch.CLOUD,
                                 ch.SUCCESSRATE,ch.ALGORITHMTYPE,ch.MP,
                                 ch.TYPE,ch.SELECTED,ch.VERSION])
            listOfValues = [np.round(x,decimals=decimals) 
                                for x in list(values.iloc[0])]
            df[ch.SUCCESSRATE] = listOfValues
            df[ch.ALGORITHMTYPE] = [ch.RANDOM,ch.LEARNED]
    
            df[ch.MP] = int(folder[-1])
            df[ch.TYPE] = df[ch.ALGORITHMTYPE].apply(
                                    lambda x:x+','+ch.MP+'='+folder[-1])
            capa = int(folder[0:re.search('-',folder).span()[0]])
            df[ch.CAPACITY] = capa * 2
            df[ch.STANDARD] = capa
            df[ch.CLOUD] = capa
            df[ch.SELECTED] = 'Y'
            df[ch.VERSION] = 'v2'
            df = df.loc[~df[ch.SUCCESSRATE].isnull()]
            
            if len(servicedata)==0:
                servicedata = df
            else:
                servicedata = pd.concat([servicedata,df],axis=0)
    
    servicedata[ch.FAILEDRATE] = 1 - servicedata[ch.SUCCESSRATE]
    servicedata.to_csv(filepath,sep=';',index=None)
    return servicedata
    

def outputServiceLevel(filename='servicelevel_rebid.csv',
                       separate=None,size=(8,4),ci=None,vertical=True,
                       legendFontsize=None,tickFontsize=None,path=LOGS_DIR):
    graph = Graphics(path)
    dataAll = pd.read_csv(os.path.join(path,filename),sep=';')
    dataAll = dataAll[dataAll[ch.SELECTED]=='Y']
    dataAll = _changeColumnName(dataAll,ch.MAXREBIDOLD,ch.MP)
    
    if separate is None:
        hue = ch.TYPE
        data = dataAll[[ch.CAPACITY,ch.SUCCESSRATE,hue]]
        data1 = dataAll[[ch.CAPACITY,ch.FAILEDRATE,hue]]
    else:
        hue = ch.TYPE
        if separate==ch.MAXREBIDOLD:
            separate = ch.MP
        data = dataAll[[ch.CAPACITY,ch.SUCCESSRATE,separate,hue]]
        data1 = dataAll[[ch.CAPACITY,ch.FAILEDRATE,separate,hue]]
    
    if ci is None:
        data = data.groupby([ch.CAPACITY,separate,hue],
                            as_index=False).mean().round(3)
        data1 = data1.groupby([ch.CAPACITY,separate,hue],
                            as_index=False).mean().round(3)
    
    graph._drawLineplot(data,x=ch.CAPACITY,y=ch.SUCCESSRATE,
          title='Success rate with different resource capacity',
          hue=hue,style=hue,order='sorted',legends=4,
          legendFontsize=legendFontsize,tickFontsize=tickFontsize,
          size=size,separate=separate,ci=ci,
          vertical=vertical)#,showTable=True)
    graph._drawLineplot(data1,x=ch.CAPACITY,y=ch.FAILEDRATE,
          title='Failure rate with different resource capacity',
          hue=hue,style=hue,order='sorted',legends='best',
          legendFontsize=legendFontsize,tickFontsize=tickFontsize,
          size=size,separate=separate,ci=ci,
          vertical=vertical,yscale='log',yaxisTick='right')#,showTable=True)

def outputUtilizationOverTimeLowResCapa(path=LOGS_DIR,nrSites='2sites',
                                site='site0',service='service2',
                                legends=None,stepRange=None,avgTime=1,
                                avgMax=False,ci=None):
    if path is None:
        path = os.path.join(path,'logs_multipleService',nrSites)
    graph = Graphics(path)
    data = graph._collectData('performance')    
    data = data.loc[data[ch.NRSITES]==nrSites]
    if stepRange is None:
        stepRange = (min(data[ch.STEP]),max(data[ch.STEP]))    
    data = data.loc[(data[ch.STEP]<=stepRange[1]) 
                        & (data[ch.STEP]>=stepRange[0])]
    if avgTime>1:
        data[ch.STEP] = data[ch.STEP].apply(lambda x: int(x/avgTime)*avgTime)

    data[ch.TRAINED] = np.where(data[ch.TRAINED]==ch.RANDOM,ch.RANDOM,
                                                                ch.LEARNED)
    data[ch.MODELTYPE] = np.where(data[ch.MODELTYPE]=='ConvCritic_ConvActor',
                                 'CNN-HW','MLP')        
    data[ch.MODELTYPE] = np.where(data[ch.TRAINED]==ch.RANDOM,
                                 '',data[ch.MODELTYPE])
    data = data[data[ch.MODELTYPE]!='MLP']

    pattern = re.compile('cloud|standard|slow')
    sitetype = pattern.findall(data[ch.SITETYPE].iloc[0])
    sitetype = dict([('site'+str(i),x) 
                        for i,x in enumerate(sitetype)])
    
    utilized = graph._parseColumn(data,ch.UTILIZATION,sitetype)
    df = pd.concat([data,utilized],axis=1)
    analyzeCols = [x for x in utilized.columns if site in x and service in x]
    
    try:
        df[ch.CAT] = df[ch.TRAINED]+' '+df[ch.INTERVAL]
    except:
        df[ch.CAT] = df[ch.TRAINED]
    hue = ch.CAT
    hue_order = list(set(df[ch.CAT]))
    hue_order.sort(reverse=False)
    
    if avgMax and ci is None:
        charttype = 'max'
    else:
        charttype = 'mean'

    for t in analyzeCols:
        if ci is None:
            if avgMax:
                tmp = df[[ch.STEP,hue,t]].groupby(
                                    [ch.STEP,hue],as_index=False).max()
            else:
                tmp = df[[ch.STEP,hue,t]].groupby(
                                    [ch.STEP,hue],as_index=False).mean()
        else:
            tmp = df[[ch.STEP,hue,t]]
        tmp.columns = [ch.STEP,hue,ch.UTILIZATION]
        graph._drawLineplot(df=tmp,x=ch.STEP,y=ch.UTILIZATION,
            title=t+'_'+charttype, style=hue,hue=hue,
            hue_order=hue_order,legends=legends,
            tickFontsize=12,ci=ci,ylim=(0,1))
        

def outputNrRebidData(subfolder,resource,name,path=LOGS_DIR):
    path = os.path.join(path,subfolder)
    graph = Graphics(path)
    
    data = graph._collectData(name)
    if data.shape[0]<=1:
        return
    
    col = [ch.STEP,ch.BATCHCREATE,ch.CREATETIME,ch.FINISHTIME,ch.NRREBID]
    for c in col:
        data[c] = data[c].astype(int)
    
    col = [ch.BIDPRICE,ch.BIDPAYMENT,ch.CARBUDGET]
    for c in col:
        data[c] = data[c].astype(float)            
    
    df = data.copy()
    df[ch.VEHICLE] = df[ch.BIDID].apply(lambda x: 
                                    x[0:re.search('_',x).span()[0]])
    
    maxStep = max(df[ch.STEP])
    df1 = df[df[ch.STEP]>maxStep-5000]
    # average staging time
    df1[ch.BACKOFFTIME] = df1.apply(lambda row: 
                        row[ch.CREATETIME]-row[ch.BATCHCREATE],axis=1)
    
    # correlation admission rate vs. price vs. staging time, by qos
    df1[ch.ADMITTED] = np.where(df1[ch.STATUS]=='finished',True,False)
    df1 = _changeColumnName(df1,ch.NRSITES,ch.MAXREBIDOLD2)
    df1[ch.MAXREBIDOLD2] = df1[ch.MAXREBIDOLD2].apply(lambda x: int(x[0]))
    df1[ch.TRAINED] = np.where(df1[ch.TRAINED]==ch.RANDOM,ch.RANDOM,
                               df1[ch.TRAINED])
    
    ttl = df1.groupby([ch.MAXREBIDOLD2,ch.TRAINED,ch.VEHICLE,ch.BIDQOS,
                       ch.CARBUDGET],as_index=False)\
        .agg({ch.BIDID:'size'}).rename(columns={ch.BIDID:ch.TOTALBYQOS})
    df1 = pd.merge(df1,ttl,on=[ch.MAXREBIDOLD2,ch.TRAINED,ch.VEHICLE,ch.BIDQOS,
                               ch.CARBUDGET],copy=False)
    df2 = df1[[ch.MAXREBIDOLD2,ch.TRAINED,ch.VEHICLE,ch.ADMITTED,ch.BIDQOS,
               ch.BIDID,ch.BIDPRICE,ch.BACKOFFTIME,ch.TOTALBYQOS,
               ch.CARBUDGET,ch.NRREBID]].groupby([ch.MAXREBIDOLD2,ch.TRAINED,
               ch.VEHICLE,ch.ADMITTED,ch.BIDQOS,ch.CARBUDGET],
               as_index=False).agg({ch.BIDID:'size',ch.BIDPRICE:'mean',
               ch.BACKOFFTIME:'mean',ch.TOTALBYQOS:'mean',ch.NRREBID:'mean'})
    df2.sort_values(ch.TRAINED,ascending=False,inplace=True)
    df2[ch.CAPACITY] = resource
    return df2

def outputNrRebidComparisonBoxplot(subfolder='25-25-60',path=LOGS_DIR):
    name = 'finishedBids'
    resource = int(subfolder[0:2]) * 2
    df2 = outputNrRebidData(subfolder=subfolder,resource=resource,name=name,
                            path=path)
    title = (name+'_distr-nrRebid_all_byAlgorithm')
    graph = Graphics(path)
    graph._drawBoxplot(df=df2,x=ch.MAXREBIDOLD2,y=ch.NRREBID,title=title,
            hue=ch.TRAINED,ylabel='rebidding overhead',legends=2,
            legendFontsize=8,figsize=(2,4))


def _drawHistLine(path,data_learned,data_random,note,
                  col=None,xlim=(0,1),loc=4,size=(10,4)):
    if col is None:
        vehicleSuccess = data_learned.groupby(ch.VEHICLE).agg(
                {ch.BIDID:'size',ch.SUCCESS:'sum'})
        tmp = data_random.groupby(ch.VEHICLE).agg(
                {ch.BIDID:'size',ch.SUCCESS:'sum'})
    else:
        vehicleSuccess = data_learned.groupby([ch.VEHICLE,col],
                        as_index=False).agg({ch.BIDID:'size',ch.SUCCESS:'sum'})
        tmp = data_random.groupby([ch.VEHICLE,col],as_index=False).agg(
                {ch.BIDID:'size',ch.SUCCESS:'sum'})
    vehicleSuccess[ch.SUCCESSRATE] = (vehicleSuccess[ch.SUCCESS] 
                                        / vehicleSuccess[ch.BIDID])
    vehicleSuccess[ch.TRAINED] = ch.LEARNED
    
    tmp[ch.SUCCESSRATE] = tmp[ch.SUCCESS] / tmp[ch.BIDID]
    tmp[ch.TRAINED] = ch.RANDOM
    
    vehicleSuccess = pd.concat([vehicleSuccess,tmp])
    style = {ch.LEARNED:'-',ch.RANDOM:'--'}
    graph = Graphics(path)
    graph._drawCdfFromKde(df=vehicleSuccess,hue=ch.TRAINED,
        target=ch.SUCCESSRATE,style=style,
        title='allocation_'+note+'_vehicleSuccessRateCdf',
        xlim=xlim,col=col,loc=loc,size=size)
    
    vehicleFailure = vehicleSuccess
    vehicleFailure[ch.FAILEDRATE] = 1 - vehicleFailure[ch.SUCCESSRATE]
    graph._drawCdfFromKde(df=vehicleFailure,hue=ch.TRAINED,
        target=ch.FAILEDRATE,style=style,
        title='allocation_'+note+'_vehicleFailureRateCdf',
        xlim=(1-xlim[1],1-xlim[0]),col=col,loc=loc,size=size)

def outputIndividualSuccessRateWithHighResCapa(path=LOGS_DIR,
                                    folder='35-35-60-rebid=1',size=(10,4)):
    path = os.path.join(path,folder)
    graph = Graphics(path)
    name = 'finishedBids'
    data = graph._collectData(name)
    data[ch.MODELTYPE] = 'CNN-HW'
    data = data[data[ch.STEP]>7000]
    col = [ch.STEP,ch.SUCCESS,ch.BATCHCREATE,ch.CREATETIME,ch.FINISHTIME,
           ch.NRREBID]
    for c in col:
        try:
            data[c] = data[c].astype(int)
        except:
            pass
    target = ch.BIDPRICE
    col = [target,ch.BIDPAYMENT,ch.CARBUDGET]
    for c in col:
        data[c] = data[c].astype(float)
    data[ch.VEHICLE] = data[ch.BIDID].apply(
            lambda x:x[0:re.search('_',x).span()[0]])
    data[ch.CARBUDGET] = np.where(data[ch.CARBUDGET]<2000,'low','high')
    data = _changeColumnName(data,ch.CARBUDGET,ch.BUDGET)
    
    data_learned = data[data[ch.TRAINED]==ch.LEARNED]
    data_random = data[data[ch.TRAINED]==ch.RANDOM]
    _drawHistLine(path,data_learned,data_random,
                  'highResCapa_budgets',col=ch.BUDGET,xlim=(0.6,1),loc=4,
                  size=size)

def _outputUtilizationMeanAndStdHighResCapa_perCapa(path=LOGS_DIR,
                                        folder='35-35-60-rebid=1',capa=35):
    path = os.path.join(path,folder)
    graph = Graphics(path)
    name = 'performance'
    data = graph._collectData(name)
    if len(data)==0:
        return None,None
    data[ch.TRAINED] = np.where(data[ch.TRAINED]==ch.RANDOM,
                                ch.RANDOM,data[ch.TRAINED])
    pattern = re.compile('cloud|standard|slow')
    sitetype = pattern.findall(data[ch.SITETYPE].iloc[0])
    sitetype = dict([('site'+str(i),x) 
                        for i,x in enumerate(sitetype)])
    
    utilized = graph._parseColumn(data,ch.UTILIZATION,sitetype)
    occupied = graph._parseColumn(data,ch.OCCUPIED,sitetype)
    maxAmount = graph._parseColumn(data,ch.MAXAMOUNT,sitetype)
    
    data = pd.concat([data,utilized,occupied,maxAmount],axis=1)
    df0 = data[data[ch.STEP]>5000]
    df0.fillna(0,inplace=True)
    
    targets = []
    for s in sitetype.keys():
        targets.append(s)
        occupiedColn = [x for x in df0.columns if ch.OCCUPIED in x 
                        and s in x and 'resource' not in x]
        maxAmountColn = [x for x in df0.columns if ch.MAXAMOUNT in x 
                         and s in x and 'resource' not in x]
        df0[s] = df0.apply(lambda row: 
            sum(row[occupiedColn])/sum(row[maxAmountColn]),axis=1)
        
    colname = 'allsites'
    targets.append(colname)
    occupiedColn = list(occupied.columns)
    maxAmountColn = [x for x in maxAmount.columns if 'resource' not in x]
    df0[colname] = df0.apply(lambda row: 
        sum(row[occupiedColn])/sum(row[maxAmountColn]),axis=1)
    
    def myFunc(data):
        results = []
        colnames = []
        for c in targets:
            results += [np.mean(data[c]),np.std(data[c])]
            colnames += [c+'_mean', c+'_std']
        return pd.Series(data=results,index=colnames)
    tmp = df0[[ch.TRAINED]+targets].groupby(ch.TRAINED).apply(myFunc)
    tmp.sort_index(inplace=True,ascending=False)
    tmp = tmp.transpose()
    tmp.reset_index(inplace=True)
    tmp.columns = [ch.SITE,ch.LEARNED,ch.RANDOM]
    tmp[ch.REBID] = int(path[-1])
    tmp[ch.CAPACITY] = capa
    df0[ch.REBID] = int(path[-1])
    df0[ch.CAPACITY] = capa * 2
    return df0,tmp

def outputUtilizationMeanAndStdHighResCapa(boxchart=None,path=LOGS_DIR,
                                           rebid='5',target='site0'):
    if boxchart is None:
        folders = []
        for i in np.arange(25,120,step=5):
            folders.append(
                    '-'.join([str(int(i)),str(int(i)),'60','rebid='])+rebid)
        boxchart = pd.DataFrame()
        for folder in folders:
            capa = int(folder[0:re.search('-',folder).span()[0]])
            df0,tmp = _outputUtilizationMeanAndStdHighResCapa_perCapa(
                                            path=path,folder=folder,capa=capa)
            if df0 is None:
                continue
            if len(boxchart)==0:
                boxchart = df0
            else:
                boxchart = pd.concat([boxchart,df0],axis=0)
#        boxchart = _changeColumnName(boxchart,'resource',ch.CAPACITY)

    graph = Graphics(path)
    graph._drawBoxplot(df=boxchart,x=ch.CAPACITY,
           y=target,ylabel='utilization',hue=ch.TRAINED,
           title='utilization_boxplot_'+target+'_rebid='+rebid,
           legendFontsize=12,figsize=(6,4),legends=1)
    return boxchart

def outputReliability(path=LOGS_DIR,rebid=['1','5'],stepRange=(0,8000)):
    name = 'finishedBids'
    folders = []
    capas = []
    for r in rebid:
        for i in np.arange(25,120,step=5):
            folders.append(
                    '-'.join([str(int(i)),str(int(i)),'60','rebid='])+r)
            capas.append((int(i),r))
    
    colnames = [ch.TRAINED,ch.CAPACITY,ch.MP,ch.ADMITTED,ch.RELIABILITY]
    result = []
    for i,folder in enumerate(folders):
        path0 = os.path.join(path,folder)
        graph = Graphics(path0)
        data = graph._collectData(name)
        if len(data)==0:
            continue
        data = data[(data[ch.STEP]>=stepRange[0]) 
                        & (data[ch.STEP]<stepRange[1])]
        for alg in set(data[ch.TRAINED]):
            admitted = data[(data[ch.STATUS]=='finished') &
                            (data[ch.TRAINED]==alg)]
            successful = admitted[admitted[ch.SUCCESS]==1]
            reliability = successful.shape[0] / admitted.shape[0]
            result.append(dict(zip(colnames,[alg,capas[i][0],capas[i][1],
                                             admitted.shape[0],reliability])))
    return pd.DataFrame(result)        

def outputRebidBoxplotMeanAndStd(boxchart=None,path=LOGS_DIR,rebid='5'): 
    if boxchart is None:
        name = 'finishedBids'
        folders = []
        for i in np.arange(25,120,step=5):
            folders.append(
                    '-'.join([str(int(i)),str(int(i)),'60','rebid='])+rebid)
        
        boxchart = pd.DataFrame()
        for folder in folders:
            resource = int(folder[0:re.search('-',folder).span()[0]]) * 2
            df0 = outputNrRebidData(subfolder=folder,resource=resource,
                                    name=name,path=path)
            if df0 is None:
                continue
            if len(boxchart)==0:
                boxchart = df0
            else:
                boxchart = pd.concat([boxchart,df0],axis=0)
    
    graph = Graphics(path)
    graph._drawBoxplot(df=boxchart,x=ch.CAPACITY,y=ch.NRREBID,
                       ylabel='rebidding overhead',hue=ch.TRAINED,
                       title='rebid='+rebid+'_boxplot',
                       legendFontsize=8,figsize=(6,4),legends=None)
    boxchart[ch.MAXREBIDOLD2] = rebid
    return boxchart

def getInterval(tbl,targetCol,rebidCol,algorithmCol,
                capaCol=ch.CAPACITY):
    interval = pd.DataFrame()
    tbl[rebidCol] = tbl[rebidCol].apply(str)
    for rebid in ['1','5']:
        for capa in set(tbl[capaCol]):
            x1 = list(tbl[(tbl[rebidCol]==rebid)
                        & (tbl[capaCol]==capa) 
                        & (tbl[algorithmCol]==ch.LEARNED)][targetCol])
            x2 = list(tbl[(tbl[rebidCol]==rebid)
                        & (tbl[capaCol]==capa) 
                        & (tbl[algorithmCol]==ch.RANDOM)][targetCol])
            if len(x1)==0:
                continue
            meanDiff,interval_ttest,interval_welch = welch_ttest(x1,x2)
            tmp = pd.DataFrame(
                [[rebid,capa,meanDiff,interval_ttest,interval_welch]],
                columns=[ch.MP,ch.CAPACITY,'mean difference',
                     'confidence interval ttest','confidence interval welch'])
            if len(interval)==0:
                interval = tmp
            else:
                interval = pd.concat([interval,tmp],axis=0)
    interval.sort_values(by=[ch.MP,ch.CAPACITY],inplace=True)
    return interval

def outputComparisonTbl(tbl,targetCol,rebidCol=ch.MP,
                        capaCol=ch.CAPACITY,algorithmCol=ch.TRAINED):
    tbl_1 = tbl[[algorithmCol,capaCol,targetCol]].groupby(
            [algorithmCol,capaCol],as_index=False).mean()
    tbl_2 = tbl_1.pivot_table(values=targetCol,index=capaCol,
                              columns=algorithmCol)
    tbl_2[ch.DIFFERENCE] = (tbl_2[ch.RANDOM] 
                                - tbl_2[ch.LEARNED]) / tbl_2[ch.RANDOM]
    interval = getInterval(tbl,targetCol=targetCol,rebidCol=rebidCol,
                           algorithmCol=algorithmCol,capaCol=capaCol)
    return tbl_2, interval

def findOffloadRate(tbl,target,rebid=1,minRate=None,maxRate=None,
                    rebidCol=ch.MP):
    result = None
    try:
        if minRate is not None:
            result = min(tbl.loc[(tbl[rebidCol]==rebid) 
                                   & (tbl[target]>=minRate),ch.CAPACITY])
        elif maxRate is not None:
            result = max(tbl.loc[(tbl[rebidCol]==rebid)
                                   & (tbl[target]<=maxRate),ch.CAPACITY])
        return result
    except: 
        return result

def outputServiceCmpTbl(servicedata):
    serviceComp = servicedata[[ch.ALGORITHMTYPE,ch.MP,
                    ch.CAPACITY,ch.SUCCESSRATE]].pivot_table(
                    index=[ch.MP,ch.CAPACITY],
                    columns=ch.ALGORITHMTYPE,values=ch.SUCCESSRATE)
    serviceComp[ch.DIFFERENCE] = (serviceComp[ch.RANDOM]
                        - serviceComp[ch.LEARNED]) / serviceComp[ch.RANDOM]
    for c in serviceComp.columns:
        serviceComp[c] = serviceComp[c].apply(lambda x: np.round(x,2))
    serviceComp.reset_index(inplace=True)
    offloadRate = {}
    for rebid in [1,5]:
        for minRate in [0.98,0.99]:
            for target in [ch.LEARNED,ch.RANDOM]:
                offloadRate[(rebid,minRate,target)] = findOffloadRate(
                        serviceComp,target=target,rebid=rebid,minRate=minRate)
    
    interval = getInterval(servicedata,targetCol=ch.SUCCESSRATE,
                           rebidCol=ch.MP,algorithmCol=ch.ALGORITHMTYPE)
    return serviceComp, offloadRate, interval

def outputBackoffChartsComparison(rebid='1',contention=None,figsize=(10,4),
                                  path=LOGS_DIR):
    if contention=='high':
        folders = []
        for i in np.arange(25,50,step=5):
            folders.append('-'.join([str(int(i)),str(int(i)),
                                     '60','rebid='])+rebid)
    elif contention=='low':
        folders = []
        for i in np.arange(55,120,step=5):
            folders.append('-'.join([str(int(i)),str(int(i)),
                                     '60','rebid='])+rebid)        
    else:
        contention = 'all'
        folders = []
        for i in np.arange(25,120,step=5):
            folders.append('-'.join([str(int(i)),str(int(i)),
                                     '60','rebid='])+rebid)

    backoffBudget = pd.DataFrame()
    for folder in folders:
        backoffData = outputBackOffCharts(subdir=[folder],outputGraph=False,
                                          path=path)
        if len(backoffBudget)==0:
            backoffBudget = backoffData
        else:
            backoffBudget = pd.concat([backoffBudget,backoffData],axis=0)
    
    for dl in ['long deadline','short deadline']:
        tmp = backoffBudget.loc[backoffBudget[ch.DEADLINE]==dl]
        graph = Graphics(path)
        graph._drawBoxplot(df=tmp,x=ch.CAPACITY,y=ch.BACKOFFTIME,
           ylabel=ch.BACKOFFTIME,hue=ch.PRICERANGE,
           title='backoffBudget_'+contention+'Contention_'+dl +'_rebid='+rebid,
           legendFontsize=14,figsize=figsize,legends=1,
           myPalette={'low price':'C1','high price':'C0'}) 
        
    return backoffBudget

def welch_ttest(x1,x2,ci=0.95,tail='one'):
    if tail=='two':
        ci = 1 - (1-ci)/2    
    n1 = len(x1)
    n2 = len(x2)
    mu1 = np.mean(x1)
    mu2 = np.mean(x2)
    dof1 = n1-1
    dof2 = n2-1
    var1 = np.var(x1,ddof=1)
    var2 = np.var(x2,ddof=1)
    pooled_samplevar = (dof1 * var1 + dof2 * var2) / (dof1 + dof2)
    pooled_sd = np.sqrt(pooled_samplevar)
    t1 = tDistr.ppf(ci,dof1+dof2)
    interval_ttest = t1 * pooled_sd * np.sqrt(1/n1 + 1/n2)
    
    welch_dof = (var1/n1 + var2/n2)**2 / ( 
                            (var1/n1)**2 / dof1 + (var2/n2)**2 / dof2 )
    t2 = tDistr.ppf(ci,welch_dof)
    interval_welch = t2 * np.sqrt(var1/n1 + var2/n2)
    meanDiff = mu1 - mu2
    return meanDiff,interval_ttest,interval_welch

def _collectRewardData(folder,filename,columnName,dataRange,
                       targetCol=ch.AVGREWARD,path=LOGS_DIR):
    if folder is None or folder=='':
        filepath = os.path.join(path,filename)
    else:
        filepath = os.path.join(path,folder,filename)
    try:
        data = pd.read_csv(filepath,sep=';')
    except:
        return
    data1 = data[[ch.STEP,targetCol]].groupby(
            ch.STEP,as_index=False).mean()
    data1 = data1[(data1[ch.STEP]<=dataRange[1]) 
                   & (data1[ch.STEP]>dataRange[0])]
    data1.columns = [ch.STEP,columnName]
    return data1

def outputCuriosityReward(filepath=LOGS_DIR,folder='',dataRange=(100,2000),
                          interval=None):
    if interval is None:
        interval = ''
    else:
        interval = '_' + str(interval)
    filename = 'invModel' + interval + '_2_True.txt'
    dataInv = _collectRewardData(folder=folder,filename=filename,
                                 targetCol=ch.AVGINVLOSS,
                                 columnName=ch.AVERAGELOSS,
                                 dataRange=dataRange)
    filename = 'forwardModel' + interval + '_2_True.txt'
    dataFwd = _collectRewardData(folder=folder,filename=filename,
                                 targetCol=ch.AVGFWDLOSS,
                                 columnName=ch.AVERAGELOSS,                                 
                                 dataRange=dataRange)
    dataInv_out = dataInv.copy()
    dataFwd_out = dataFwd.copy()
    dataInv_out.columns = [ch.STEP,ch.AVERAGELOSSINVERSE]
    dataFwd_out.columns = [ch.STEP,ch.AVERAGELOSSFORWARD]
    
    avgStep = 100
    dataInv[ch.STEP] = dataInv[ch.STEP].apply(lambda x: int(x/avgStep)*avgStep)
    dataFwd[ch.STEP] = dataFwd[ch.STEP].apply(lambda x: int(x/avgStep)*avgStep)
    
    graph = Graphics(filepath)
    graph._drawLineplot(df=dataInv,x=ch.STEP,y=ch.AVERAGELOSS,
                        title='avgLoss_inverse'+interval,legends=None,
                        legendFontsize=None,tickFontsize=12,
                        size=None,decimals=2,ci='sd',showTable=False,
                        ylim=(0,None))
    graph._drawLineplot(df=dataFwd,x=ch.STEP,y=ch.AVERAGELOSS,
                        title='avgLoss_forward'+interval,legends=None,
                        legendFontsize=None,tickFontsize=12,
                        size=None,decimals=2,ci='sd',showTable=False,
                        ylim=(0,None))
    return (dataInv_out,dataFwd_out)

def outputFspReward(filepath=LOGS_DIR,folder='30-30-60-rebid=1',
                    dataRange=(100,1500),interval=None,const=None):
    graph = Graphics(filepath)
    if interval is None:
        interval = ''
    else:
        interval = '_' + str(interval)
    dataSL = pd.DataFrame()
    filelist = [''] + list(np.arange(0,10))
    for i in filelist:
        filename = 'supervisedLearningModel'+str(i)+interval+'_2_True.txt'
        data_sl = _collectRewardData(folder=folder,filename=filename,
                            columnName=ch.AVERAGELOSSSL,
                            dataRange=dataRange)
        if data_sl is None:
            continue
        if len(dataSL)==0:
            dataSL = data_sl
        else:
            dataSL = pd.concat([dataSL,data_sl],axis=0)

    avgStepAtt = 1000
    avgStepExt = 1000
    avgStepInt = 500
    dataInt = pd.DataFrame()
    dataExt = pd.DataFrame()
    dataAtt = pd.DataFrame()
    for i in filelist:
        try:
            filename = 'reward'+str(i)+interval+'_2_True.txt'
            data_int = _collectRewardData(folder=folder,filename=filename,
                                columnName=ch.INTRINSICREWARD,
                                targetCol=ch.INTREWARD,
                                dataRange=dataRange)
            data_ext = _collectRewardData(folder=folder,filename=filename,
                                columnName=ch.EXTRINSICREWARD,
                                targetCol=ch.EXTREWARD,
                                dataRange=dataRange)
            if data_int is not None:
                if len(dataInt)==0:
                    dataInt = data_int
                else:
                    dataInt = pd.concat([dataInt,data_int],axis=0)
            if data_ext is not None:
                if len(dataExt)==0:
                    dataExt = data_ext
                else:
                    dataExt = pd.concat([dataExt,data_ext],axis=0)
        except:
            pass

        filename = 'priceLearningModel'+str(i)+interval+'_2_True.txt'
        try:
            data_att = _collectRewardData(folder=folder,filename=filename,
                            columnName=ch.AVERAGEATTENTIONLOSS,
                            targetCol=ch.ATTENTIONLOSS,
                            dataRange=(dataRange[0],1000000))
        except:
            continue
        if data_att is not None:
            if len(dataAtt)==0:
                dataAtt = data_att
            else:
                dataAtt =pd.concat([dataAtt,data_att],axis=0)
    if len(dataAtt)>0:
        dataAtt[ch.STEP] = dataAtt[ch.STEP].apply(lambda x: 
                                                int(x/avgStepAtt)*avgStepAtt)
        graph._drawLineplot(df=dataAtt,x=ch.STEP,y=ch.AVERAGEATTENTIONLOSS,
                            title='avgAttentionLoss'+interval,legends=None,
                            legendFontsize=None,tickFontsize=12,
                            size=None,decimals=2,ci='sd',showTable=False) 
        
    try:
        data = dataSL.merge(dataInt,on=ch.STEP,how='outer')
    except:
        filename = 'priceLearningModel'+interval+'_2_True.txt'
        dataRL = _collectRewardData(folder=folder,filename=filename,
                                columnName=ch.AVERAGEREWARDRL,
                                dataRange=dataRange)
        data = dataSL.merge(dataRL,on=ch.STEP,how='outer')
        return data,dataAtt
    data = data.merge(dataExt,on=ch.STEP,how='outer')
    data.fillna(0,inplace=True)
    data_out = data.copy()
    if len(dataAtt>0):
        dataAtt = dataAtt[~dataAtt[ch.AVERAGEATTENTIONLOSS].isnull()]

    if const is None:
        const = pmp.fsp_rl_weight_constant
    data[ch.WEIGHTEDINTRINSICREWARD] = (
            (1-data[ch.AVERAGELOSSSL]) * (1-const/(data[ch.STEP]+const)) 
            + const/(data[ch.STEP]+const) * data[ch.INTRINSICREWARD])
    
    data1 = data.copy()
    data[ch.STEP] = data[ch.STEP].apply(lambda x: int(x/avgStepInt)*avgStepInt)
    
    graph._drawLineplot(df=data,x=ch.STEP,y=ch.WEIGHTEDINTRINSICREWARD,
                        title='weightIntReward'+interval,legends=None,
                        legendFontsize=None,tickFontsize=12,
                        size=None,decimals=2,ci='sd',showTable=False)
    graph._drawLineplot(df=data,x=ch.STEP,y=ch.INTRINSICREWARD,
                        title='avgIntReward'+interval,legends=None,
                        legendFontsize=None,tickFontsize=12,
                        size=None,decimals=2,ci='sd',showTable=False)
    graph._drawLineplot(df=data,x=ch.STEP,y=ch.AVERAGELOSSSL,
                        title='avgLoss_sl'+interval,legends=None,
                        legendFontsize=None,tickFontsize=12,
                        size=None,decimals=2,ci='sd',showTable=False)

    data1[ch.STEP] = data1[ch.STEP].apply(
                                    lambda x:int(x/avgStepExt)*avgStepExt)
    graph._drawLineplot(df=data1,x=ch.STEP,y=ch.EXTRINSICREWARD,
                        title='avgExtReward'+interval,legends=None,
                        legendFontsize=None,tickFontsize=12,
                        size=None,decimals=2,ci='sd',showTable=False)
    
    return (data_out,dataAtt)

def addRows(tbl,dataRange,valueCol,dataCol,filler):
    exist = set(tbl[dataCol])
    diff = list(set(dataRange)-exist)
    exist = list(exist)
    exist.sort()
    for i in diff:
        oneSmaller = exist[0]
        oneBigger = i
        for j in exist:
            if j < i and j > oneSmaller:
                oneSmaller = j
            if j > i:
                oneBigger = j
                break
        resBefore = tbl[tbl[dataCol]==oneSmaller][valueCol].tolist()[-1]
        resAfter = tbl[tbl[dataCol]==oneBigger][valueCol].tolist()[0]
        newRes = int(np.round((resAfter - resBefore) * (i - oneSmaller) 
                        / (oneBigger - oneSmaller) + resBefore,decimals=0))
        tmpPd = pd.DataFrame([filler + [newRes,i]],columns=tbl.columns)
        tbl = pd.concat([tbl,tmpPd],axis=0)
    return tbl

def outputServiceTable(serviceComp,size=None,
                       legendFontsize=None,tickFontsize=None,path=LOGS_DIR,
                       output='success'):
    if isinstance(serviceComp[ch.MP][0],str):
        serviceComp[ch.MP]=serviceComp[ch.MP].apply(int)
    dra = serviceComp[[ch.MP, ch.CAPACITY, ch.LEARNED]]
    ria = serviceComp[[ch.MP, ch.CAPACITY, ch.RANDOM]]
    perfRange = np.arange(0.90,0.985,step=0.01)
    perfRange = [np.round(x,decimals=2) for x in perfRange]
    target = ch.SUCCESSRATE
    if output=='failure':
        target = ch.FAILEDRATE

    draco = pd.DataFrame()
    rial = pd.DataFrame()
    for rebid in [1,5]:
        tmp = dra[dra[ch.MP]==rebid]
        tmp = addRows(tbl=tmp,dataRange=perfRange,
                      valueCol=ch.CAPACITY,dataCol=ch.LEARNED,
                      filler=[rebid])
        if len(draco)==0:
            draco = tmp
        else: 
            draco = pd.concat([draco,tmp],axis=0)
        
        tmp0 = ria[ria[ch.MP]==rebid]
        tmp0 = addRows(tbl=tmp0,dataRange=perfRange,
                       valueCol=ch.CAPACITY,dataCol=ch.RANDOM,
                       filler=[rebid])
        if len(rial)==0:
            rial = tmp0
        else:
            rial = pd.concat([rial,tmp0],axis=0)

    col = [ch.MP,ch.CAPACITY,target,ch.TYPE] 
    draco[ch.TYPE] = ch.LEARNED
    draco.columns = col
    rial[ch.TYPE] = ch.RANDOM
    rial.columns = col
    data = pd.concat([draco,rial],axis=0)
    data = data[(data[target]>=perfRange[0])]       
    data1 = data.groupby([ch.MP,target,ch.TYPE],as_index=False).agg('min')

    dataRange = perfRange+[0.99]
    xticklabel = None
    if output=='failure':
        xticklabel = np.array([np.round(1-x,decimals=2) for x in dataRange]) 
    
    graph = Graphics(path)
    graph.drawRegplot(df=data1,x=target,y=ch.CAPACITY,
        title='capacityUse_'+target,hue=ch.TYPE,order='fixed',size=size,
        legendFontsize=legendFontsize,tickFontsize=tickFontsize,legends='best',
        separate=ch.MP,x_decimals=2,y_decimals=0,
        dataRange=dataRange,xticklabel=xticklabel)
    
    data = data.pivot_table(values=ch.CAPACITY,
                index=[ch.MP,target],
                columns=ch.TYPE,aggfunc='min')
    data.reset_index(inplace=True)
    data[ch.SAVING] = (data[ch.RANDOM] - data[ch.LEARNED]) / data[ch.RANDOM]
    data[ch.SAVING] = data[ch.SAVING].apply(
            lambda x: np.round(x,decimals=2) if not np.isnan(x) else '')
    for col in [ch.LEARNED,ch.RANDOM]:
        data[col] = data[col].apply(
                lambda x: int(x) if not np.isnan(x) else '')
    return data,data1

def outputSumoTrace(filepath=LOGS_DIR,folder='',
                    file='',gridRangeX=(1,401),gridRangeY=(1,307),
                    sumoDataOffset=50000,nrRecord=200000):
    graph = Graphics(os.path.join(filepath,folder))
    df = graph.drawTraceDataDistr(
            file=file,gridRangeX=gridRangeX,gridRangeY=gridRangeY,
            sumoDataOffset=sumoDataOffset,nrRecord=nrRecord)
    return df

def outputAttentionReward(filepath,dataInv,dataFwd,dataRL,dataAtt,
                                  dataRange=(0,50000),interval=None):
    if interval is None:
        interval = ''
    else:
        interval = '_' + str(interval)

    invLoss_weight = cmp.invLoss_weight
    rl_weight = pmp.fsp_rl_weight_constant
    curiosity_weight = 0
    data = pd.merge(dataInv,dataFwd,how='outer',on=ch.STEP)
    data = data.merge(dataRL,how='outer',on=ch.STEP)
    data.fillna(0,inplace=True)
    data = data[(data[ch.STEP]>dataRange[0]) & (data[ch.STEP]<=dataRange[1])]
    
    data[ch.CURIOSITYLOSS] = (invLoss_weight * data[ch.AVERAGELOSSINVERSE] 
                          + (1-invLoss_weight) * data[ch.AVERAGELOSSFORWARD])
    
    data[ch.AVERAGEINTRINSICREWARD] = (
        (1-data[ch.AVERAGELOSSSL]) * (1-rl_weight/(data[ch.STEP]+rl_weight)) 
        + rl_weight/(data[ch.STEP]+rl_weight) 
          * ((1-curiosity_weight)*data[ch.INTRINSICREWARD]
             +curiosity_weight*(1-data[ch.CURIOSITYLOSS]))
    )
    
    avgStep = 50
    data[ch.STEP] = data[ch.STEP].apply(lambda x: int(x/avgStep)*avgStep)
    
    graph = Graphics(filepath)
    graph._drawLineplot(df=data,x=ch.STEP,y=ch.AVERAGEINTRINSICREWARD,
                        title='avgIntReward'+interval,legends=None,
                        legendFontsize=None,tickFontsize=12,
                        size=None,decimals=2,ci='sd',showTable=False)
    
    if len(dataAtt)>0:
        avgStep = 100
        dataAtt[ch.STEP] = dataAtt[ch.STEP].apply(lambda x: 
                                                int(x/avgStep)*avgStep)
   


#%%
if __name__ == '__main__': 
    path = LOGS_DIR
    try:
        path = os.path.join(path,sys.argv[3])
    except:
        pass

    if sys.argv[1]=='curious':
#        testDf = outputSumoTrace(file='test',
#                                 nrRecord=100000,sumoDataOffset=75000)
#        evalDf = outputSumoTrace(file='eval',
#                                 nrRecord=100000,sumoDataOffset=100000)
#        trainNewDf = outputSumoTrace(file='trainNew',
#                                 nrRecord=200000,sumoDataOffset=52000)
        
        interval = int(sys.argv[2])
        if interval>0:
#            folder = os.path.join('conext',
#                                  'attentionRNNsupervised_capa=30_train')
#            folder = os.path.join('conext',
#                                  'intrinsicRewardWithFwdloss+payoff_ascent_capa=30_train')
            folder = ''
            dataInv,dataFwd = outputCuriosityReward(filepath=path,
                    folder=folder,dataRange=(0,50000),interval=interval)
            dataRL,dataAtt = outputFspReward(filepath=path,folder=folder,
                            dataRange=(0,50000),interval=interval)       
#            outputAttentionReward(filepath=path,
#                    dataInv=dataInv,dataFwd=dataFwd,dataRL=dataRL,
#                    dataAtt=dataAtt,dataRange=(2000,50000),interval=interval)
#            outputUtilizationOverTimeLowResCapa(path=path,nrSites='2sites',
#                            site='site',service='service',
#                            stepRange=(60000,130000),
#                            avgTime=1000,ci='sd',
#                            legends='best')
    else: # general setup
        path = os.path.join(path,'general')
        folder = '30-30-60-rebid=1'
        graph = Graphics(os.path.join(path,folder))
#        graph.drawPerformance(drawPerformanceOnly=True,textOutput=False,
#                              sites='2sites',target=ch.TOTALSUCCESSRATIO,
#                              legends=4,stepRange=(300,7999),
#                              decimals=2,ci='sd')
        graph.drawPerformance(drawPerformanceOnly=True,textOutput=False,
                              sites='2sites',target=ch.TOTALSUCCESSRATIO,
                              legends='best',stepRange=(300,7999),
                              yaxisTick='right',
                              decimals=2,ci='sd')
        outputUtilizationOverTimeLowResCapa(
                              path=os.path.join(path,folder),
                              nrSites='2sites',site='site0',service='service2',
                              stepRange=(7700,7999))
#        outputFspReward(filepath=path,folder=folder,dataRange=(150,7999))
        
        decimals = 3
        filename = 'servicelevel_rebid_decimals='+str(decimals)+'.csv'
        servicedata = getServiceLevelData(filename=filename,decimals=decimals,
                                          path=path)

        size = (5, 4)
        legendFontsize = 10
        tickFontsize = 10
        outputServiceLevel(filename=filename,separate=ch.MP,ci='sd',
                           size=size,legendFontsize=legendFontsize,
                           tickFontsize=tickFontsize,vertical=True,path=path)
        serviceComp,offloadRate,serviceCi = outputServiceCmpTbl(servicedata)
        serviceTbl,tbl = outputServiceTable(serviceComp,size=size,
                        legendFontsize=legendFontsize,
                        tickFontsize=tickFontsize,path=path)
        serviceTbl,tbl = outputServiceTable(serviceComp,size=size,
                        legendFontsize=legendFontsize,
                        tickFontsize=tickFontsize,path=path,
                        output='failure')

    
        try:
            rebidTbl5 = pd.read_csv(
                                os.path.join(path,'rebid=5_Tbl.csv'),sep=';')
            rebidTbl5 = outputRebidBoxplotMeanAndStd(
                                        boxchart=rebidTbl5,path=path,rebid='5')
        except:
            rebidTbl5 = outputRebidBoxplotMeanAndStd(path=path,rebid='5')
            rebidTbl5.to_csv(os.path.join(path,'rebid=5_Tbl.csv'),
                                                         sep=';',index=None)
        rebidComp, rebidCi = outputComparisonTbl(tbl=rebidTbl5,
                        targetCol=ch.NRREBID,rebidCol=ch.MAXREBIDOLD2,
                        capaCol=ch.CAPACITY,algorithmCol=ch.TRAINED)
     
        try:
            utilTbl1 = pd.read_csv(os.path.join(path,'rebid=1_utilTbl.csv'),
                                   sep=';')
            utilTbl1 = outputUtilizationMeanAndStdHighResCapa(
                                        boxchart=utilTbl1,path=path,rebid='1')
        except:
            utilTbl1 = outputUtilizationMeanAndStdHighResCapa(path=path,
                                                              rebid='1')
            utilTbl1.to_csv(os.path.join(path,'rebid=1_utilTbl.csv'),
                            sep=';',index=None)
        
        utilComparison, utilCi = outputComparisonTbl(tbl=utilTbl1,
            targetCol='site0',rebidCol=ch.REBID,capaCol=ch.CAPACITY,
            algorithmCol=ch.TRAINED)
        
        outputIndividualSuccessRateWithHighResCapa(path=path,size=(14,4))
    
        _ = outputBackOffCharts(subdir=['25-25-60-rebid=5'],path=path,
                                size=(7,4))
        backoffBudget5 = outputBackoffChartsComparison(
                    rebid='5',contention='high',figsize=(7,4),path=path)

        testDf = outputSumoTrace(file='test',
                                 nrRecord=100000,sumoDataOffset=75000)

#        testGraph = graph.drawPerformance(drawPerformanceOnly=True,
#                      target='totalSuccessRatio',textOutput=True,
#                      outputThres=1000,endOfGame=200000,
#                      legends='best',stepRange=(25000,125000),
#                      decimals=2,yaxisTick='right',orderHue='by algorithm',
#                      density=100,ci='sd')

        evalDf = outputSumoTrace(file='eval',
                                 nrRecord=100000,sumoDataOffset=100000)

#        evalGraph = graph.drawPerformance(drawPerformanceOnly=True,
#                      target='totalSuccessRatio',textOutput=True,
#                      outputThres=1000,endOfGame=200000,
#                      legends='best',stepRange=(50000,150000),
#                      decimals=2,yaxisTick='right',orderHue='by algorithm',
#                      density=100,ci='sd')

        trainNewDf = outputSumoTrace(file='trainNew',
                                 nrRecord=200000,sumoDataOffset=52000)
        
#        trainGraph = graph.drawPerformance(drawPerformanceOnly=True,
#                      target='totalSuccessRatio',textOutput=True,
#                      outputThres=1000,endOfGame=200000,
#                      legends='best',stepRange=(2000,200000),
#                      decimals=2,yaxisTick='right',orderHue='by algorithm',
#                      density=300,ci='sd')
        
        result = outputReliability(path=path)


