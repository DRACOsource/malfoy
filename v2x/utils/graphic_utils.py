# -*- coding: utf-8 -*-
from ..config.config import (GRAPH_DIR, MDL_PARAMS as mp, 
                             RESOURCE_SITE_PARAMS as rsp,
                             VEHICLE_PARAMS as vp,
                             PRICE_MODEL_PARAMS as pmp)
from ..supports.data import TraceData
from ..utils.name_utils import ColumnHead as ch
import glob,os,re,ast
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15
plt.rc('pdf',fonttype=42)
plt.ioff()
import seaborn as sns
from scipy.stats import gaussian_kde


class OOMFormatter(ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

class Graphics:
    learned = ch.LEARNED
    random = ch.RANDOM
    retrain = ch.RETRAIN
    att = ch.ATT
    cur = ch.CUR
    
    def __init__(self,path):
        self.path = path
    
    def _collectData(self,name,density=1):
        name = name + '*'
        filename = os.path.join(self.path,name)
        perfiles = glob.glob(filename)
        data = pd.DataFrame()
        data_part = pd.DataFrame()
        for f in perfiles:
            try:
                data_part = pd.read_csv(f,sep=';')
            except:
                continue
            
            if density>1:
                data_part[ch.STEP] = data_part[ch.STEP].apply(
                        lambda x: int(x/density) * density)
                groupcols = list(data_part.columns)
                groupcols = [x for x in groupcols 
                     if data_part[x].dtype==np.object 
                     and x not in [ch.OCCUPIED,ch.UTILIZATION,ch.MAXAMOUNT]]
                data_part = data_part.groupby(
                        [ch.STEP]+groupcols, as_index=False).mean()
            
            locations = [x.span() for x in re.finditer('_',f)]
            try:
                nrSites = str(f[locations[-2][1]:locations[-1][0]]) + 'sites'
                fileversion = str(f[locations[-2][0]-1
                                              :locations[-2][0]])
                if not re.match('\d',fileversion):
                    fileversion = ''
            except:
                continue
            trained = (self.learned 
                       if f[locations[-1][1]:-4]=='True' else self.random)
            if 'att' in f.lower():
                trained = self.att
            elif 'cur' in f.lower():
                trained = self.cur
            if 'Retrain' in f:
                trained = trained + ' ' + self.retrain

            data_part[ch.NRSITES] = nrSites
            data_part[ch.TRAINED] = trained
            
            try:
                interval = str(f[locations[-3][1]:locations[-2][0]])
            except:
                interval = 0
            data_part[ch.INTERVAL] = interval
            
            if ch.BIDID in data_part.columns:
                data_part[ch.BIDID] = data_part[ch.BIDID].apply(
                                                lambda x: fileversion+x)
            
            if data.shape[0]==0:
                data = data_part
            else:
                data = pd.concat([data,data_part],axis=0)
        
        if not ch.MODELTYPE in data.columns:
            data[ch.MODELTYPE] = 'MLP'
        data[ch.BATCHSIZE] = pmp.batch_size
        return data
    
    def _drawLineplot(self,df,x,y,title,style=None,hue=None,order='flex',
                      hue_order=None,legends=2,legendFontsize=None,
                      tickFontsize=None,size=None,separate=None,
                      decimals=1,ci=None,showTable=False,vertical=True,
                      ylim=None,yscale='linear',yaxisTick='left'):
        defaultFontsize = 16
        if tickFontsize is None:
            tickFontsize = defaultFontsize
        if legendFontsize is None:
            legendFontsize = defaultFontsize
        if size is None:
            length = 5
            height = 4
        else:
            length = size[0]
            height = size[1]

        if separate is None:
            if hue is not None:
                if hue_order is None:
                    if order=='flex':
                        tmp = df.groupby(hue)
                        tmp = (tmp.tail(1).drop_duplicates()
                               .sort_values(y,ascending=False))
                    else:
                        tmp = df.groupby(
                                hue,as_index=False).max().sort_values(hue)
                    hue_order = tmp[hue].tolist()
            else:
                hue_order = None
            
            fig,ax = plt.subplots()
            fig.set_size_inches(length,height)
            
            try:
#                density = max(len(hue_order),5)
#                dashes = [(density-x, max(2,x) if x>0 else 0,
#                           2*x,       max(2,x) if x>0 else 0) 
#                                for x in list(range(len(hue_order)))]
                dashes = [(len(hue_order)-x,x,2*x,x)
                                for x in list(range(len(hue_order)))]
            except:
                dashes = None
            try:
                ax = sns.lineplot(x=x,y=y,style=style,hue=hue,data=df,ci=ci,
                                  hue_order=hue_order,style_order=hue_order,
                                  dashes=dashes)
            except ValueError: # not enough styles
                ax = sns.lineplot(x=x,y=y,style=style,hue=hue,data=df,ci=ci,
                                  hue_order=hue_order,dashes=dashes)                
                
            if ylim is not None:
                ax.set_ylim(ylim[0],ylim[1])
            ax.set_yscale(yscale)
            
            ax.set_xlabel(xlabel=x,fontsize=tickFontsize)
            ax.set_ylabel(ylabel=y,fontsize=tickFontsize)
#            ax.set_xticklabels(np.int0(ax.get_xticks()),size=tickFontsize)
 
            ax.tick_params(axis='both',which='major',labelsize=tickFontsize)
            ax.tick_params(axis='both',which='minor',labelsize=tickFontsize-3)
            yformatter = ScalarFormatter(useOffset=True,useMathText=True)
            xformatter = OOMFormatter(order=3,fformat='%2.0f')
            ax.yaxis.set_major_formatter(yformatter)
            ax.yaxis.set_minor_formatter(yformatter)
            ax.xaxis.set_major_formatter(xformatter)
            ax.xaxis.get_offset_text().set_fontsize(tickFontsize)
#            ax.ticklabel_format(style='sci', axis='x', scilimits=(3,3))            

            if decimals==0:
                ax.set_yticklabels(np.int0(ax.get_yticks()),size=tickFontsize)                
            else:
                ax.set_yticklabels(np.round(ax.get_yticks(),
                                   decimals=decimals),size=tickFontsize)  
            
            if yaxisTick=='right':
                ax.yaxis.tick_right()
                
            if legends is not None:
                handles, labels = ax.get_legend_handles_labels() 
                l = ax.legend(handles[1:],labels[1:],loc=legends,
                              fontsize=legendFontsize)
                plt.savefig(os.path.join(GRAPH_DIR,
                            title.replace(' ','')+'.pdf'),
                            bbox_extra_artists=(l,), bbox_inches='tight')
            else:
                l = ax.legend()
                l.remove()
                plt.savefig(os.path.join(GRAPH_DIR,
                            title.replace(' ','')+'.pdf'),
                            bbox_inches='tight')
            plt.clf()
        else:
            sepCol = list(set(df[separate]))
            if vertical:
                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
            else:
                fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
            fig.set_size_inches(length,height)
            
            df1 = df.loc[df[separate]==sepCol[0]]
            df2 = df.loc[df[separate]==sepCol[1]]
            
            if hue is not None:
                if hue_order is None:
                    if order=='flex':
                        tmp1 = df1.groupby(hue)
                        tmp1 = (tmp1.tail(1).drop_duplicates()
                               .sort_values(y,ascending=False))
                        tmp2 = df2.groupby(hue)
                        tmp2 = (tmp2.tail(1).drop_duplicates()
                               .sort_values(y,ascending=False))
                    else:
                        tmp1 = df1.groupby(
                                hue,as_index=False).max().sort_values(hue)
                        tmp2 = df2.groupby(
                                hue,as_index=False).max().sort_values(hue)
                    hue_order1 = tmp1[hue].tolist()
                    hue_order2 = tmp2[hue].tolist()
            else:
                hue_order1 = None
                hue_order2 = None
            
            g1 = sns.lineplot(x=x,y=y,style=style,hue=hue,data=df1,ci=ci,
                        hue_order=hue_order1,style_order=hue_order1,ax=ax1)      
            g2 = sns.lineplot(x=x,y=y,style=style,hue=hue,data=df2,ci=ci,
                        hue_order=hue_order2,style_order=hue_order2,ax=ax2)
            
            ax1.set_yscale(yscale)
            ax2.set_yscale(yscale)
            ax1.set_xticklabels(np.int0(ax1.get_xticks()),
                                size=tickFontsize)
            ax2.set_xticklabels(np.int0(ax2.get_xticks()),
                                size=tickFontsize)
            ax1.set_yticklabels(np.round(ax1.get_yticks(),
                                decimals=decimals),size=tickFontsize)
            ax2.set_yticklabels(np.round(ax2.get_yticks(),
                                decimals=decimals),size=tickFontsize)
            ax1.xaxis.label.set_size(tickFontsize)
            ax2.xaxis.label.set_size(tickFontsize)
            ax1.yaxis.label.set_size(tickFontsize)
            ax2.yaxis.label.set_size(tickFontsize)
            
            if showTable:
                fig.subplots_adjust(hspace=0.5)
                ax1.xaxis.set_visible(False)
                ax2.set_xticklabels([])
                ax2.xaxis.labelpad = 30
                ax2.tick_params(bottom=False)
                
                values1,values5,alg,column = self._createTbl(servicedata=df,
                            hue=hue,alg=[self.learned,self.random],
                            separate=separate,target=y)
                ax1.table(cellText=values1,rowLabels=alg,
                                          colLabels=column,loc='bottom')
                ax2.table(cellText=values5,rowLabels=alg,
                                          colLabels=column,loc='bottom')

            if yaxisTick=='right':
                ax1.yaxis.tick_right()
                ax2.yaxis.tick_right()
            
            if legends is not None:
                handles1, labels1 = ax1.get_legend_handles_labels() 
                ax1.legend(handles1[1:],labels1[1:],loc=legends,
                           fontsize=legendFontsize)
                handles2, labels2 = ax2.get_legend_handles_labels() 
                ax2.legend(handles2[1:],labels2[1:],loc=legends,
                           fontsize=legendFontsize)            
            else:
                l = ax.legend()
                l.remove()
            plt.savefig(os.path.join(GRAPH_DIR,title.replace(' ','')+'.pdf'),
                        bbox_inches='tight')
            plt.clf()
          
    def _drawCdfFromKde(self,df,hue,target,style,title,
                        col=None,xlim=(0,1),loc=4,size=(10,4)):
        if col is None:
            plt.figure(figsize=(5,4))
            hue_order = list(set(df[hue]))
            hue_order.sort()
            for grp in hue_order:
                tmp = df.loc[df[hue]==grp,target]
                tmp = np.array(tmp)
                kde = gaussian_kde(tmp)
                cdf = np.vectorize(lambda x: kde.integrate_box_1d(-np.inf,x))
                x = np.linspace(xlim[0],xlim[1])
                plt.plot(x,cdf(x),linestyle=style[grp],label=grp)
            plt.legend(loc=loc,fontsize=15)
            plt.ylabel('CDF',fontsize=15)
            plt.xlabel(target,fontsize=15)
            print(target)
            plt.savefig(os.path.join(GRAPH_DIR,title.replace(' ','')+'.pdf'),
                        bbox_inches='tight')
            plt.clf()
        else:
            x = np.linspace(xlim[0],xlim[1])
            newDf = pd.DataFrame()
            for c in set(df[col]):
                for grp in set(df[hue]):
                    tmp = df.loc[(df[hue]==grp) & (df[col]==c),target]
                    tmp = np.array(tmp)
                    kde = gaussian_kde(tmp)
                    cdf = np.vectorize(
                            lambda y:kde.integrate_box_1d(-np.inf,y))
                    tmp0 = pd.DataFrame(np.vstack([x,cdf(x)]).transpose(),
                                         columns=[target,'CDF'])
                    tmp0[hue] = grp
                    tmp0[col] = c
                    if len(newDf)==0:
                        newDf = tmp0                        
                    else:
                        newDf = pd.concat([newDf,tmp0],axis=0)
            fig,ax = plt.subplots()
            ax = sns.FacetGrid(data=newDf,col=col,)
            ax.fig.set_size_inches(size[0],size[1])
            ax.map_dataframe(sns.lineplot,target,'CDF',hue,
                    style=hue,hue_order=list(style.keys()),
                    style_order=list(style.keys()),ci=None)
            ax.set(xlim=xlim)
            for axes in ax.axes.flat:
                axes.set_ylabel('CDF', fontsize=15)
                axes.set_xlabel(target, fontsize=15)
                axes.set_title(axes.get_title(),fontsize=15)
            handles, labels = ax.axes[0][-1].get_legend_handles_labels()
            l = ax.axes[0][-1].legend(handles[1:],labels[1:],
                                           loc=loc,fontsize=15)
            plt.savefig(os.path.join(GRAPH_DIR,title.replace(' ','')+'.pdf'),
                        bbox_extra_artists=(l,),bbox_inches='tight')
            plt.clf()
    
    def _drawBoxplot(self,df,x,y,title,hue=None,legends=3,ylabel=None,
                     legendFontsize=None,figsize=None,
                     myPalette=None,hue_order=None):
        if figsize is None:
            figsize = (5,4)
        defaultFontsize = 16
        if legendFontsize is None:
            legendFontsize = defaultFontsize
        if ylabel is None:
            ylabel = y
        
        sns.set_style('white')
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)
        if myPalette is None:
            myPalette = {self.random:'C1',self.learned:'C0'}
        sns.boxplot(data=df,x=x,y=y,ax=ax,hue=hue,
                    showfliers=False,palette=myPalette,
            showmeans=True,meanprops={'marker':'o','markerfacecolor':'white',
                            'markeredgecolor':'white'},hue_order=hue_order)
        ax.set_xlabel(xlabel=x,fontsize=defaultFontsize)
        ax.set_ylabel(ylabel=ylabel,fontsize=defaultFontsize)
        if len(set(df[x]))>12:
            for ind, label in enumerate(ax.get_xticklabels()):
                if ind % 2 == 0:
                    label.set_visible(True)
                else:
                    label.set_visible(False)
        
        if legends is not None:
            handles, labels = ax.get_legend_handles_labels()
            l = ax.legend(handles,labels,loc=legends,fontsize=legendFontsize)
            plt.savefig(os.path.join(GRAPH_DIR,title.replace(' ','')+'.pdf'),
                    bbox_extra_artists=(l,), bbox_inches='tight')
        else:
            l = ax.legend()
            l.remove()
            plt.savefig(os.path.join(GRAPH_DIR,title.replace(' ','')+'.pdf'),
                        bbox_inches='tight')
        plt.clf()
    
    def _parse(self,value,sitetype):
        a = ast.literal_eval(value)
        values = {}
        for i,x in enumerate(a):
            site = 'site' + str(i)
            stype = sitetype[site]
            for key in x.keys():
                values[stype+'_'+site+'_'+str(key[1])] = x[key]
        return pd.Series(values)
    
    def _parseColumn(self,df,target,sitetype):
        result = df[target].apply(lambda x: self._parse(x,sitetype))
        col = result.columns
        col0 = [target + '_' + x for x in col]
        result.columns = col0
        return result


    def _outputFailureComparison(self,data=None,path=None,prefix='',
                textOutput=False,graphicOutput=True,legends=2,stepRange=None,
                nrSites='2sites'):
        name = 'performance'
        if data is None:
            data = self._collectData(name)
        try:
            data[ch.SUCCESSRATE] = data[ch.SUCCESS]/data[ch.FINISHEDBID]
        except:
            data[ch.SUCCESSRATE] = data[ch.SUCCESS]/data[ch.TOTALBID]
        try:
            data[ch.FAILEDRATE] = data[ch.REJECTEDBID]/data[ch.FINISHEDBID]
        except:
            data[ch.FAILEDRATE] = data[ch.REJECTEDBID]/data[ch.TOTALBID]
#            data[ch.TRAINED] = np.where(data[ch.TRAINED]==self.random,
#                                        self.random,self.learned)
        data[ch.CATEGORY] = data.apply(lambda row: 
                    row[ch.TRAINED] + '_' + str(row[ch.NRSITES]) 
                    + '_' + str(row[ch.MODELTYPE]),axis=1)
        cols = list(data.columns)
        if ch.SUCCESSRATEBYTIME not in cols:
            cols[cols.index(ch.TOTALSUCCESSRATIO)] = ch.SUCCESSRATEBYTIME
            data.columns = cols
        
        barChart = pd.DataFrame()
        coln = [ch.NRSITES,'success '+self.learned,
                'success '+self.random,
                ch.SUCCESSRATE,'failed '+self.learned,
                'failed '+self.random,ch.FAILEDRATE]
        barGrp = ['failed '+self.random, 'failed '+self.learned]
        barGrp2 = ['success '+self.random,'success '+self.learned]
        note = 'learned_vs_random'

        learned = data[(data.trained==self.learned) & (data.nrSites==nrSites)]
        random = data[(data.trained==self.random) & (data.nrSites==nrSites)]
        try:
            min_learned = max(learned[ch.STEP]) - mp.recent
        except:
            min_learned = 0
        try:
            min_random = max(random[ch.STEP]) - mp.recent
        except:
            min_random = 0
        learned = learned.loc[learned[ch.STEP]>=min_learned,:]
        random = random.loc[random[ch.STEP]>=min_random,:]
        
        successLearned = np.mean(learned[ch.SUCCESSRATE])
        successRandom = np.mean(random[ch.SUCCESSRATE])
        successRate = (successLearned-successRandom)/successRandom
        failedLearned = np.mean(learned[ch.FAILEDRATE])
        failedRandom = np.mean(random[ch.FAILEDRATE])
        failedRate = -(failedLearned-failedRandom)/failedRandom
        row = pd.DataFrame([[nrSites,successLearned,successRandom,successRate,
                failedLearned,failedRandom,failedRate]],columns=coln)
        if len(barChart)==0:
            barChart = row
        else:
            barChart = pd.concat([barChart,row],axis=0)
        
        if stepRange is None:
            if max(data.step)>4000:
                data = data[data.step>800]
        else:
            data = data.loc[(data[ch.STEP]<=stepRange[1]) 
                                & (data[ch.STEP]>=stepRange[0])]
        xGrp = ch.NRSITES
        barChart.sort_values(by=xGrp,inplace=True)
        data = data[[ch.STEP,ch.MODELTYPE,ch.TRAINED,ch.SUCCESSRATE]].groupby(
                    [ch.STEP,ch.MODELTYPE,ch.TRAINED],as_index=False).mean()
        if graphicOutput:
            self._drawLineplot(df=data,x=ch.STEP,y=ch.SUCCESSRATE,
                title=name+'_line_'+prefix+'_success rate_'+note,
                style=ch.TRAINED,hue=ch.TRAINED,order='flex',legends=legends)
            self._drawBarChart(df=barChart,xGrp=xGrp,barGrp=barGrp2,
                               yLabel=ch.SUCCESSRATE,
                               title=name+'_'+prefix+'_success rate_'+note)
            self._drawBarChart(df=barChart,xGrp=xGrp,barGrp=barGrp,
                               yLabel='failed rate',
                               title=name+'_'+prefix+'_failed rate_'+note)
        if textOutput:
            print('capa:{},target:{},value:{}'.format(
                    rsp.serviceCapa,ch.SUCCESSRATE,barChart[barGrp2]))
        else:
            return(barChart[barGrp2])

    def _createTbl(self,servicedata,hue,alg=None,separate=ch.MAXREBIDOLD,
                   target=ch.SUCCESSRATE):    
        servicedata[separate] = servicedata[separate].apply(str)
        if alg is None:
            alg = list(set(servicedata[ch.ALGORITHMTYPE]))
        alg.sort(reverse=False)
        rebid = list(set(servicedata[separate]))
        rebid.sort()
        
        row1 = [x+','+ch.MP+'='+rebid[0] for x in alg]
        row5 = [x+','+ch.MP+'='+rebid[1] for x in alg]
        column = np.arange(
            min(servicedata[servicedata[separate]==rebid[0]][ch.CAPACITY]),
            max(servicedata[servicedata[separate]==rebid[0]][ch.CAPACITY])+1,
            10,dtype=int).tolist()
        
        servicedata = servicedata[[ch.CAPACITY,hue,target]]
        for capa,algorithm in product(column,row1+row5):
            tmp = pd.DataFrame([[int(capa),algorithm,np.nan]],
                                 columns=servicedata.columns)
            servicedata = pd.concat([servicedata,tmp],axis=0)
        
        data = servicedata[[ch.CAPACITY,hue,target]].groupby(
                [ch.CAPACITY,hue],as_index=False).mean().round(3)
        data.sort_values(by=ch.CAPACITY,ascending=True,inplace=True)
        
        values1 = []
        values5 = []
        for r in row1:
            values = data[data[ch.TYPE]==r][target].tolist()
            values0 = [np.round(x,decimals=2) if not np.isnan(x) 
                                                    else '' for x in values]
            values1.append(values0)
        for r in row5:
            values = data[data[ch.TYPE]==r][target].tolist()
            values0 = [np.round(x,decimals=2) if not np.isnan(x) 
                                                    else '' for x in values]
            values5.append(values0)
        
        return values1,values5,alg,column

    def _getPerformanceData(self,name,stepRange,sites,
                            extRewardInterval='all',density=1):
        data = self._collectData(name,density)        
        if stepRange is None:
            stepRange = (min(data[ch.STEP]),max(data[ch.STEP]))
        try:
            data = data.loc[(data[ch.STEP]<=stepRange[1]) 
                            & (data[ch.STEP]>=stepRange[0])]
        except:
            try:
                stepRange = (min(data[ch.STEP]),max(data[ch.STEP]))
            except:
                return
            data = data.loc[(data[ch.STEP]<=stepRange[1]) 
                            & (data[ch.STEP]>=stepRange[0])]
        if len(data)==0:
            return
        
        data[ch.SUCCESSRATE] = data[ch.SUCCESS] / data[ch.FINISHEDBID]                
        data[ch.MODELTYPE] = np.where(
                data[ch.MODELTYPE]=='ConvCritic_ConvActor','CNN-HW','MLP')        
        data[ch.MODELTYPE] = np.where(data[ch.TRAINED]==self.random,
                                     '',data[ch.MODELTYPE])
        try:
            data[ch.CATEGORY] = data.apply(lambda row: 
                        row[ch.TRAINED] + '_' + str(row[ch.NRSITES])
                        + '_' + str(row[ch.EXTREWARDINTERVAL])
                        + '_' + str(row[ch.MODELTYPE]),axis=1)            
        except:
            data[ch.CATEGORY] = data.apply(lambda row: 
                        row[ch.TRAINED] + '_' + str(row[ch.NRSITES]) 
                        + '_' + str(row[ch.MODELTYPE]),axis=1)
        cols = list(data.columns)
        cols[cols.index(ch.TOTALSUCCESSRATIO)] = ch.SUCCESSRATEBYTIME
        data.columns = cols
        if sites is None:
            sites = set(data[ch.NRSITES])
        data = data.loc[data[ch.NRSITES].isin(sites)]
        if isinstance(extRewardInterval,str):
            try:
                extRewardInterval = set(data[ch.EXTREWARDINTERVAL])
            except:
                data[ch.EXTREWARDINTERVAL] = vp.curiosityExtRewardThres
                extRewardInterval = [vp.curiosityExtRewardThres]
            try:
                data = data.loc[data[ch.EXTREWARDINTERVAL].isin(extRewardInterval)]
            except:
                pass
        return data
    
    def _getRegData(self,data,x,y):
        model = lambda x,a1,a2,a3,a4,a5,a6,a7: a1+a2*x+a3*x**2+a4*x**3+a5*x**4      
        mdl = model
        a,b = curve_fit(mdl,data[x],data[y])
        lst = np.array(data[x])
        pts = mdl(lst,*a)
        return pts
    
    def drawRegplot(self,df,x,y,title,style=None,hue=None,order='flex',
                      hue_order=None,legends=2,legendFontsize=None,
                      tickFontsize=None,size=None,separate=None,
                      x_decimals=1,y_decimals=1,linestyle=None,
                      dataRange=None,xticklabel=None):
        defaultFontsize = 15
        if tickFontsize is None:
            tickFontsize = defaultFontsize
        if legendFontsize is None:
            legendFontsize = defaultFontsize
        if size is None:
            length = 5
            height = 4
        else:
            length = size[0]
            height = size[1]
        if linestyle is None:
            linestyle = ['-','--']
        if dataRange is not None:
            try:
                df = df[(df[x]>=min(dataRange)) & 
                        (df[x]<=max(dataRange))]
            except:
                pass

        if separate is None:
            if hue is not None:
                if hue_order is None:
                    if order=='flex':
                        tmp = df.groupby(hue)
                        tmp = (tmp.tail(1).drop_duplicates()
                               .sort_values(y,ascending=False))
                    else:
                        tmp = df.groupby(
                                hue,as_index=False).max().sort_values(hue)
                    hue_order = tmp[hue].tolist()
            else:
                hue_order = None
            
            fig,ax = plt.subplots()
            fig.set_size_inches(length,height)
            
            for i,h in enumerate(hue_order):
                tmp = df[df[hue]==h]
                regData = self._getRegData(tmp,x,y)
                ax.scatter(x=tmp[x].values,y=tmp[y].values)
                ax.plot(tmp[x].values,regData,label=h,linestyle=linestyle[i]) 
        
            ax.set_xlabel(xlabel=x,fontsize=tickFontsize)
            ax.set_ylabel(ylabel=y,fontsize=tickFontsize)
            if dataRange is not None:
                ax.set_xticks(dataRange[0::2])
            if xticklabel is not None:
                xticklabel = xticklabel[0::2]
                ax.set_xticklabels(xticklabel)
            else:
                xticklabel = ax.get_xticks()
            
            if tickFontsize!=defaultFontsize:
                if x_decimals>0:
                    ax.set_xticklabels(np.round(xticklabel,
                                                decimals=x_decimals),
                                       size=tickFontsize)
                else:
                    ax.set_xticklabels(np.int0(xticklabel),
                                       size=tickFontsize)
                if y_decimals>0:
                    ax.set_yticklabels(np.round(ax.get_yticks(),
                                                decimals=x_decimals),
                                       size=tickFontsize)
                else:
                    ax.set_yticklabels(np.int0(ax.get_yticks()),
                                       size=tickFontsize)
            if legends is not None:
                handles, labels = ax.get_legend_handles_labels() 
                l = ax.legend(handles[0:],labels[0:],loc=legends,
                              fontsize=legendFontsize)
                plt.savefig(os.path.join(GRAPH_DIR,
                            title.replace(' ','')+'.pdf'),
                            bbox_extra_artists=(l,), bbox_inches='tight')
            else:
                l = ax.legend()
                l.remove()
                plt.savefig(os.path.join(GRAPH_DIR,
                            title.replace(' ','')+'.pdf'),
                            bbox_inches='tight')
            plt.clf()
        else:
            sepCol = list(set(df[separate]))
            sepCol.sort()
            
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
            fig.set_size_inches(length,height)
            
            df1 = df.loc[df[separate]==sepCol[0]]
            df2 = df.loc[df[separate]==sepCol[1]]
            
            if hue is not None:
                if hue_order is None:
                    if order=='flex':
                        tmp1 = df1.groupby(hue)
                        tmp1 = (tmp1.tail(1).drop_duplicates()
                               .sort_values(y,ascending=False))
                        tmp2 = df2.groupby(hue)
                        tmp2 = (tmp2.tail(1).drop_duplicates()
                               .sort_values(y,ascending=False))
                    else:
                        tmp1 = df1.groupby(
                                hue,as_index=False).max().sort_values(hue)
                        tmp2 = df2.groupby(
                                hue,as_index=False).max().sort_values(hue)
                    hue_order1 = tmp1[hue].tolist()
                    hue_order2 = tmp2[hue].tolist()
            else:
                hue_order1 = None
                hue_order2 = None
            
            for i,sep in enumerate(sepCol):
                if i==0:
                    for j,h in enumerate(hue_order1):
                        tmp = df[(df[separate]==sep) & (df[hue]==h)]
                        regData = self._getRegData(tmp,x,y)
                        ax1.scatter(x=tmp[x].values,y=tmp[y].values,s=4)
                        ax1.plot(tmp[x].values,regData,
                                 label=h+', '+separate+'='+str(sep),
                                 linestyle=linestyle[j])
                else:
                    for j,h in enumerate(hue_order2):
                        tmp = df[(df[separate]==sep) & (df[hue]==h)]
                        regData = self._getRegData(tmp,x,y)
                        ax2.scatter(x=tmp[x].values,y=tmp[y].values,s=4)
                        ax2.plot(tmp[x].values,regData,
                                 label=h+', '+separate+'='+str(sep),
                                 linestyle=linestyle[j])

            ax1.set_ylabel(ylabel=y,fontsize=tickFontsize)
            ax2.set_ylabel(ylabel=y,fontsize=tickFontsize)
            
            ax2.set_xlabel(xlabel=x,fontsize=tickFontsize)
            if dataRange is not None:
                ax2.set_xticks(dataRange[0::2])
            if xticklabel is not None:
                xticklabel = xticklabel[0::2]
                ax2.set_xticklabels(xticklabel)
            else:
                xticklabel = ax2.get_xticks()
            
            if tickFontsize!=defaultFontsize:
                if x_decimals>0:
                    ax1.set_xticklabels(np.round(xticklabel,
                                                 decimals=x_decimals),
                                        size=tickFontsize)
                    ax2.set_xticklabels(np.round(xticklabel,
                                                 decimals=x_decimals),
                                        size=tickFontsize)
                else:
                    ax1.set_xticklabels(np.int0(xticklabel),
                                        size=tickFontsize)
                    ax2.set_xticklabels(np.int0(xticklabel),
                                        size=tickFontsize)
                if y_decimals>0:
                    ax1.set_yticklabels(np.round(ax1.get_yticks(),
                                                 decimals=x_decimals),
                                        size=tickFontsize)
                    ax2.set_yticklabels(np.round(ax2.get_yticks(),
                                                 decimals=x_decimals),
                                        size=tickFontsize)
                else:
                    ax1.set_yticklabels(np.int0(ax1.get_yticks()),
                                        size=tickFontsize)
                    ax2.set_yticklabels(np.int0(ax2.get_yticks()),
                                        size=tickFontsize)

                ax1.xaxis.label.set_size(tickFontsize)
                ax2.xaxis.label.set_size(tickFontsize)
                ax1.yaxis.label.set_size(tickFontsize)
                ax2.yaxis.label.set_size(tickFontsize)
            
#            if showTable:
#                fig.subplots_adjust(hspace=0.5)
#                ax1.xaxis.set_visible(False)
#                ax2.set_xticklabels([])
#                ax2.xaxis.labelpad = 30
#                ax2.tick_params(bottom=False)
#                
#                values1,values5,alg,column = self._createTbl(servicedata=df,
#                            hue=hue,alg=['DRACO','RIAL'],separate=separate)
#                ax1.table(cellText=values1,rowLabels=alg,
#                                          colLabels=column,loc='bottom')
#                ax2.table(cellText=values5,rowLabels=alg,
#                                          colLabels=column,loc='bottom')
            
            if legends is not None:
                handles1, labels1 = ax1.get_legend_handles_labels() 
                ax1.legend(handles1[0:],labels1[0:],loc=legends,
                           fontsize=legendFontsize)
                handles2, labels2 = ax2.get_legend_handles_labels() 
                ax2.legend(handles2[0:],labels2[0:],loc=legends,
                           fontsize=legendFontsize)            
            else:
                l = ax.legend()
                l.remove()
            plt.savefig(os.path.join(GRAPH_DIR,title.replace(' ','')+'.pdf'),
                        bbox_inches='tight')
            plt.clf()


    def drawPerformance(self,name='performance',drawPerformanceOnly=True,
                        target=ch.RECENTNONREJECTRATIO,prefix='',
                        textOutput=False,sites=None,legends=None,
                        stepRange=None,
                        decimals=1,ci=None,extRewardInterval='all',
                        outputThres=None,endOfGame=None,
                        yaxisTick='left',orderHue=None,density=1):
        if outputThres is None:
            outputThres = 1000
        if sites is not None:
            sites = [sites]
        else:
            sites = ['2sites']
        if extRewardInterval is None:
            extRewardInterval = [vp.curiosityExtRewardThres]
        elif isinstance(extRewardInterval,int):
            extRewardInterval = [extRewardInterval]
            
        if target==ch.TOTALSUCCESSRATIO:
            target = ch.SUCCESSRATEBYTIME
        data = self._getPerformanceData(name,stepRange,
                                        sites,extRewardInterval,density)
        if data is None:
            return
        
        data = data[~data[target].isnull()]
        try:
            data[ch.CAT] = data[ch.TRAINED]+' '+data[ch.INTERVAL]
        except:
            data[ch.CAT] = data[ch.TRAINED]
        
        extRewardInterval = list(set(data[ch.EXTREWARDINTERVAL]))
        
        if not drawPerformanceOnly:
            pattern = re.compile('cloud|standard|slow')
            
            for nrSites in sites:
                df = data.loc[data[ch.NRSITES]==nrSites]
                sitetype = pattern.findall(df.sitetype.iloc[0])
                sitetype = dict([('site'+str(i),x) 
                                    for i,x in enumerate(sitetype)])
                
                utilized = self._parseColumn(df,ch.UTILIZATION,sitetype)
                df = pd.concat([df,utilized],axis=1)
                analyzeCols = utilized.columns
                
                for t in analyzeCols:
                    tmp = df[[ch.STEP,ch.TRAINED,t]].groupby(
                                    [ch.STEP,ch.TRAINED],as_index=False).mean()
                    self._drawLineplot(df=tmp,x=ch.STEP,y=t,
                        title=t+'_'+nrSites, style=ch.TRAINED,hue=ch.TRAINED,
                        order='fix',legends=None)
        
        if ( (len(sites)==1 and len(extRewardInterval)==1) 
             or (orderHue=='by algorithm') ):
            hue = ch.TRAINED
        else:
            hue = ch.CAT
        if ci is None:
            data0 = data[[ch.STEP,hue,target]].groupby(
                            [ch.STEP,hue],as_index=False).mean()
        else:
            data0 = data
        
        offset = min(data0[ch.STEP])
        data0[ch.STEP] = data0[ch.STEP] - offset
        
        hue_order = None
        if orderHue=='by algorithm':
            hue_retrain = [x for x in set(data0[hue].tolist()) 
                                    if self.learned+' '+self.retrain in x]
            hue_draco = [x for x in set(data0[hue].tolist()) 
                                    if self.learned in x]
            hue_rial = [x for x in set(data0[hue].tolist()) if self.random in x]
            hue_att = [x for x in set(data0[hue].tolist()) if self.att in x]
            hue_cur = [x for x in set(data0[hue].tolist()) if self.cur in x]
            hue_retrain.sort(reverse=True)
            hue_draco.sort(reverse=True)
            hue_order = hue_rial + hue_draco + hue_cur + hue_att + hue_retrain
            
        # if target==ch.SUCCESSRATEBYTIME: excludes admitted but failed bids
        self._drawLineplot(df=data0,x=ch.STEP,y=target,
                       title=name+'_' + target,
                       style=hue,hue=hue,order='fixed',legends=legends,
                       legendFontsize=10,tickFontsize=10,
                       decimals=decimals,ci=ci,yaxisTick=yaxisTick,
                       hue_order=hue_order)
        
        if target==ch.SUCCESSRATEBYTIME:
            data[ch.FAILURERATEBYTIME] = 1- data[target]
            data[ch.ADMITTEDRATEBYTIME] = data[ch.ADMITTEDRATE]
            data[ch.REJECTEDRATEBYTIME] = 1 - data[ch.ADMITTEDRATEBYTIME]
#            # admitted, both successful and failed execution, double counted if bid is admitted more than one time
#            self._drawLineplot(df=data,x=ch.STEP,y=ch.ADMITTEDRATEBYTIME,
#                           title=name+'_'+ch.ADMITTEDRATEBYTIME,
#                           style=hue,hue=hue,order='flex',legends=legends,
#                           legendFontsize=10,tickFontsize=10,
#                           decimals=decimals,ci=ci,
#                           yaxisTick=yaxisTick,hue_order=hue_order)            
        
            if hue_order is not None:
                hue_order.reverse()
            # rejected + admitted but failed
            self._drawLineplot(df=data,x=ch.STEP,y=ch.FAILURERATEBYTIME,
                           title=name+'_'+ch.FAILURERATEBYTIME,
                           style=hue,hue=hue,order='fixed',legends=legends,
                           legendFontsize=10,tickFontsize=10,
                           decimals=decimals,ci=ci,yscale='log',
                           yaxisTick=yaxisTick,hue_order=hue_order)
#            # rejected
#            self._drawLineplot(df=data,x=ch.STEP,y=ch.REJECTEDRATEBYTIME,
#                           title=name+'_'+ch.REJECTEDRATEBYTIME,
#                           style=hue,hue=hue,order='flex',legends=legends,
#                           legendFontsize=10,tickFontsize=10,
#                           decimals=decimals,ci=ci,yscale='log',
#                           yaxisTick=yaxisTick,hue_order=hue_order)
        
        if textOutput:
            eogDataAll = None
            for target in [ch.SUCCESSRATEBYTIME,ch.FAILURERATEBYTIME]:
                eogData = None
                print('\n{} :\n'.format(target))
                for cat in set(data[hue]):
                    tmp = data.loc[(data[hue]==cat),[ch.STEP,target]]
                    tmp.columns = [ch.STEP,'result']
                    tmp1 = np.mean(
                        tmp.loc[tmp[ch.STEP]>=max(tmp[ch.STEP])-outputThres,
                                'result'])
                    print('{} : {:.3f}'.format(cat,tmp1))
                    
                    maxStep = max(tmp[ch.STEP])
                    if endOfGame is None or maxStep < endOfGame:
                        outputRange = list(range(maxStep-outputThres,maxStep))
                    else:
                        outputRange = []
                        for i in range(int(maxStep / endOfGame)):
                            eog = (i + 1) * endOfGame                        
                            outputRange += list(range(eog-outputThres,eog))
                    tmp[hue] = cat
                    if eogData is None:
                        eogData = tmp.loc[tmp[ch.STEP].isin(outputRange)]
                    else:
                        eogData = pd.concat([
                            eogData,tmp.loc[tmp[ch.STEP].isin(outputRange)]],
                            axis=0)
                eogData[ch.TYPE] = target
                if eogDataAll is None:
                    eogDataAll = eogData
                else:
                    eogDataAll = pd.concat([eogDataAll,eogData],axis=0)
            result = eogDataAll.groupby([ch.TYPE,hue],as_index=False).agg(
                                                    {'result':['mean','std']})
                
            return (eogDataAll,result)
        else:
            return

    
    def drawPriceModelLoss(self,name='priceLearningModel'):
        data = self._collectData(name)
        data = data.loc[(~data[ch.ACTORLOSS].isnull()) & (data[ch.ACTORLOSS]<3) 
                        & (data[ch.ACTORLOSS]>-3)]
            
        for target in [ch.AVGREWARD,ch.CRITICLOSS,ch.ACTORLOSS]:
            title = name + '_' + target
            df = data[[ch.STEP,ch.NRSITES,target]].groupby(
                            [ch.STEP,ch.NRSITES],as_index=False).mean()
            self._drawLineplot(df=df,x=ch.STEP,y=target,
                               title=title,style=ch.NRSITES,hue=ch.NRSITES,
                               order='fix')  
            
    def drawCompetitorModelLoss(self,name='competitorLearningModel'):
        target = ch.COMPETITORLOSS
        data = self._collectData(name)
        data = data.loc[~data[target].isnull()]
        df = data.loc[data[ch.STEP]>=10]
        df = df[[ch.STEP,ch.NRSITES,target]].groupby(
                            [ch.STEP,ch.NRSITES],as_index=False).mean()
        outlier = np.percentile(df[target],80)
        df = df[df[target]<outlier]
        self._drawLineplot(df=df,x=ch.STEP,y=target,title=name,
                           style=ch.NRSITES,hue=ch.NRSITES,order='fix')

    def drawTraceDataDistr(self,file='',gridRangeX=(1,401),gridRangeY=(1,307),
                           sumoDataOffset=50000,nrRecord=200000):
        if 'eval' in file or 'test' in file or 'trainNew' in file:
            file = '_' + file
        else:
            file = ''

        filename = 'sumoTrace_schwantalerhoehe' + file + '.xml'
        
        trace = TraceData(path=os.path.join(self.path,'vanetdata'),
                          filename=filename,gridRangeX=gridRangeX,
                          gridRangeY=gridRangeY,dataOffset=sumoDataOffset)
        
        trace.readTraceDataFromSumo(nrRecord=nrRecord,posOffset=True)
        
        df = trace.sumoGrpData.copy()
        df.reset_index(inplace=True)
        df.columns = [ch.STEP,ch.VEHICLECOUNT]
        
        df[ch.VEHICLECOUNT].hist()
        plt.savefig(os.path.join(GRAPH_DIR,'vehicleCountDistr'+file+'.pdf'))
        
        self._drawLineplot(df=df,x=ch.STEP,y=ch.VEHICLECOUNT,
                               title='vehicleCountByTime'+file,
                               decimals=0,legends=None,
                               legendFontsize=12,tickFontsize=12)
        return df
