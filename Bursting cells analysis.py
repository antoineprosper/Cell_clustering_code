# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path
import fnmatch
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf #The autocorrelation function
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

#Extract the list of cells from the text file
listofcells = pd.read_csv(r'A:\Internship\Spike data\Simonnet_Brecht_2019\CSVexportFromDatabase2020\000_listofcells.txt',\
                        delim_whitespace=True)
    
#Get the path to the data that needs to be treated
data_folder_path = Path('A:\Internship\Spike data\Simonnet_Brecht_2019\CSVexportFromDatabase2020')

#Create a list of Path classes pointing to the files in the directory specified above. (Ignore subdirectories)
data_files_path = [file for file in data_folder_path.iterdir() if file.is_file()]

#Keeping track of the ID of the cells before loading them into variables
Index = [file.name[:4] for file in data_files_path if fnmatch.fnmatch(file.name, '*_times_amplitude_pkahpratio_midampwidth .txt')]

cells_data = pd.Series(data = None, index=Index, dtype = object)

#Loading the cell data into variables callable by their cell number
for f in data_files_path:
    if fnmatch.fnmatch(f.name, '*_times_amplitude_pkahpratio_midampwidth .txt'):
        cells_data[f.name[:4]] = pd.read_csv(f, delim_whitespace=True)
    
#Now that all the cells have been loaded, we need to filter out the ones that cannot be used for analysis.
#We want to remove the putative neurons from the data and the index
for i in np.where(listofcells['Group'] == 'PutativeInterneuron')[0]:
    cells_data = cells_data.drop(labels=[str(listofcells['ID'][i])])
    Index.remove(str(listofcells['ID'][i]))


       
#This dataframe will contain the spike time vectors for each cell, the ISI for each spike in these cells, and a boolean variable for
#each spike indicating if it is in a burst or not (threshold for burst is 6ms from either neighboring spikes).
cells_spike_times = pd.DataFrame(data = None, columns = ['Spike_times', 'ISIs','Burst'], index=Index, dtype=object)

#Threshold for spikes belonging to a burst in this area of the brain, here 6ms
burst_threshold = 6e-3

not_moving = pd.Series([np.where(cells_data[i]['animalspeedmorethan3cmpersec'] == 0) for i in Index], index= Index)



for i in Index:
#We want to remove the spikes recorded when the animal was not moving at least 3cm/s before loading them into the data vectors
    cells_data[i] = cells_data[i].drop(not_moving[i][0])
    
#We need to re index the columns to avoid holes in the indexing
    cells_data[i] = cells_data[i].reset_index(drop=True)
    
    cells_spike_times.at[i, 'Spike_times'] = cells_data[i]['time_sec'] #Exctract the spikes times for each cell
    

    
    
#Compute the ISI for each spike after the first to know what the interval from this to the previous one is.  
    cells_spike_times.at[i, 'ISIs'] = pd.Series([None]).append\
        (pd.Series(cells_spike_times.at[i,'Spike_times'][c] - cells_spike_times.at[i,'Spike_times'][c-1]\
                   for c in range(1,cells_spike_times.at[i,'Spike_times'].size)), ignore_index = True)
    
#Check if each spike belongs to a burst
    cells_spike_times.at[i, 'Burst'] = pd.Series([cells_spike_times.at[i,'ISIs'][1]<=burst_threshold]).append\
        (pd.Series(cells_spike_times.at[i, 'ISIs'][c] <= burst_threshold or cells_spike_times.at[i, 'ISIs'][c+1] <= burst_threshold\
         for c in range(1,cells_spike_times.at[i, 'ISIs'].size-1)).append(pd.Series\
        (cells_spike_times.at[i,'ISIs'][cells_spike_times.at[i, 'ISIs'].size-1]<=burst_threshold), ignore_index=True), ignore_index=True)
      

#Define the bursts, IBI = Intra Burst Interval : the mean ISI during one burst (cell IBI = mean of the IBI for each burst)
bursts = pd.DataFrame(data = None, columns = ['Number of bursts', 'Bursts', 'Mean spike count','Proportion of bursts w/ > 2 spikes'\
                                              ,'IBI'], index = Index)



##########################  
for i in Index:
    cell_bursts = np.where(cells_spike_times.at[i,'Burst'] == True) #Get the indices of spikes belonging to a burst
    burst_list = pd.DataFrame(data = None, columns = ['Spike count','IBI','Time stamp'], dtype=object)
    
    burst_count = 0
    burstID = 0
    for b in cell_bursts[0]:
        spike_count = 0
        
        if cells_spike_times.at[i,'Burst'][b] and (cells_spike_times.at[i,'ISIs'][b]==None or cells_spike_times.at[i,'ISIs'][b] > burst_threshold):
            burst_count+=1
            spike_count+=1
            c=b+1
            burst_ISI = 0
            burst_time_stamp = cells_spike_times.at[i,'Spike_times'][b]
            
            while c < len(cells_spike_times.at[i,'Burst'].index) and cells_spike_times.at[i,'Burst'][c] and \
                cells_spike_times.at[i,'ISIs'][c] <= burst_threshold:
                burst_ISI+=cells_spike_times.at[i,'ISIs'][c]
                spike_count+=1
                c+=1
            
            burst_list.at[burstID,'Spike count']=spike_count
            burst_list.at[burstID,'IBI']=burst_ISI/(spike_count-1)
            burst_list.at[burstID,'Time stamp'] = burst_time_stamp
            burstID+=1
    
    bursts.at[i,'Bursts'] = burst_list
    bursts.at[i,'Number of bursts'] = len(burst_list.index)
    bursts.at[i,'Mean spike count'] = burst_list.loc[:,'Spike count'].mean()
    
    if True in burst_list.gt(2,axis='Spike count').loc[:,'Spike count'].value_counts():        
        bursts.at[i, 'Proportion of bursts w/ > 2 spikes'] = \
            burst_list.gt(2,axis='Spike count').loc[:,'Spike count'].value_counts()[True]/bursts.at[i,'Number of bursts']
    else:
        bursts.at[i, 'Proportion of bursts w/ > 2 spikes'] = 0
        
    bursts.at[i, 'IBI'] = burst_list.loc[:,'IBI'].mean()
    
del b,c,i,burstID,burst_time_stamp,burst_ISI,burst_count,f,burst_list,spike_count,cell_bursts #Cleaning up the temporary variables
    

#We now want to compute the log of the ISIs to plot them into a histogram
log_binning = np.linspace(np.log(0.001),np.log(10), 60) #This will be the bins to model the ISI distribution for the graph

#Computing the log of the ISIs for each cell
logISI = pd.DataFrame(data=None, columns= Index, dtype = float)

for i in Index:
    temp = pd.Series(np.log(cells_spike_times.at[i,'ISIs'][c]) for c in range(1,len(cells_spike_times.at[i,'ISIs'])))
    if temp.size > logISI[i].size:
        logISI = logISI.reindex(temp.index)
    logISI[i] = temp


#We need to keep the values of the histogram somewhere:
logISI_hist= pd.DataFrame(data=None,dtype=float, columns=Index)
autoCorr = pd.DataFrame(data=None, dtype=float,columns=Index)

fig1, axISI = plt.subplots() #Plot for the ISI distrib
fig1.suptitle('Distribution of the log(ISI) for each cell', fontsize=24)
fig1.set_size_inches(15,9, forward=True)

fig2, axAcorr = plt.subplots() #Plot for the autocorrelograms
fig2.suptitle('Autocorrelation plot for the spike times of each cell', fontsize=24)
fig2.set_size_inches(15,9,forward = True)

for i in Index:
    w = np.ones(logISI[i].count())/logISI[i].count()
    freq, binEdges = np.histogram(logISI[i][0:logISI[i].count()],weights=w, bins=log_binning)
    #Note: freq is the values of the histogram : 59 of them, one for each bin
    #binEdges is the edges of the bins, 60 of them since each bin has 2 sides
    logISI_hist[i] = freq
    #Let's create a proper point for each value: the center of the bin it represents
    plotPoints = (binEdges[1:]+binEdges[:-1])/2
    
    
    #Calculate the autocorrelation values
    binningAcorr = np.arange(min(cells_spike_times.at[i,'Spike_times']), max(cells_spike_times.at[i,'Spike_times']+0.001),0.001)
    binnedSpikeTimes = np.histogram(cells_spike_times.at[i,'Spike_times'], bins= binningAcorr)
    lags_sec = np.linspace(0.001,0.020,20)
    autoCorr[i] = acf(binnedSpikeTimes[0],nlags=21,fft=False)[1:21]
    
    #Plot the two lines
    axISI.plot(plotPoints, freq, linewidth=0.5, color='black')
    
    axAcorr.plot(lags_sec, acf(binnedSpikeTimes[0],nlags=21,fft=False)[1:21], linewidth = 0.5, color='black')
    

    

axISI.set_xlabel('Inter Spike Interval (sec)', fontsize = 24)
axISI.set_ylabel('Frequency', fontsize = 24)
axISI.set_xlim(np.log(0.001),2)

axISI.tick_params(labelsize=20)
axISI.set_xticks([np.log(0.001), np.log(0.01), np.log(0.1), np.log(1)])
axISI.set_xticklabels([0.001, 0.01, 0.1, 1], fontsize = 20)
axISI.axvline(np.log(0.006), linestyle ='--', color='red')

axAcorr.set_xlabel('Lag applied for correlation (sec)', fontsize = 24)
axAcorr.set_ylabel('Autocorrelation', fontsize = 24)

axAcorr.tick_params(labelsize=20)
axAcorr.set_xticks([0.005,0.010,0.015,0.020])
axAcorr.set_xticklabels([0.005,0.01,0.015,0.02])
axAcorr.axvline(0.006, linestyle ='--', color='red')



#Create the PCA function we need
pca_ISI = PCA(n_components = 10)
pca_ISI.fit(logISI_hist) #Apply the PCA to the histograms of the logISI, going from 60 dimensions (bins) to 10.
logISI_PCs = pca_ISI.components_ #The principal component decomposition of each cell onto the 10 PC found by the function
logISI_index = ['logISI_PC1','logISI_PC2','logISI_PC3']

pca_autoCorr = PCA(n_components = 5)
pca_autoCorr.fit(autoCorr)
autoCorr_PCs = pca_autoCorr.components_
autoCorr_index = ['autoCorr_PC1','autoCorr_PC2']

#The dataframe we will use to store the projection of the cells onto the first 3 principal components of logISI and 2 of autocorrs
logISIhist_PC123_ACorr_PC12 = pd.DataFrame(logISI_PCs[0:3], dtype=float,columns=Index,index=logISI_index).append\
                                    (pd.DataFrame(autoCorr_PCs[0:2],columns=Index,index = autoCorr_index))

#We now need to transpose it because having cells as indices is easier to use.
logISIhist_PC123_ACorr_PC12 = logISIhist_PC123_ACorr_PC12.transpose()


fig3,(axWard,axISI_ordered) = plt.subplots(1,2)
fig3.set_size_inches(20,15)


#Create the clusters for the dendrogram:
clustered_hist = sch.linkage(logISIhist_PC123_ACorr_PC12,method='ward')
#Plot the dendrogram to see the clusters visually
dendFromPCA = sch.dendrogram(clustered_hist, orientation = 'left', no_labels = True, distance_sort='ascending', ax=axWard) 
#The order of the original data points is contained in dendFromPCA['leaves']
#Notice, it starts from the bottom of the tree, so for our purpose we need it backwards

#Let us try to organize the data to follow the dendrogram
#First, we get the index in the order that the dendrogram sorted them to
Index_orderedFromDend = [Index[i] for i in reversed(dendFromPCA['leaves'])] 
#Create the new dataframe to hold the sorted data
logISI_orderedFromDend = pd.DataFrame(data=None, index=Index_orderedFromDend, columns = plotPoints, dtype = float)

#Now let's fill it
for i_o in Index_orderedFromDend:
    logISI_orderedFromDend.loc[i_o][:] = logISI_hist.loc[:,i_o]

#Note : This is why we need to reverse the order from the leafs: the function plots the matrix the way it is displayed as a table:
#top left value in the top left of the image
axISI_ordered.imshow(logISI_orderedFromDend, cmap='Greys', extent = [np.log(0.001),max(logISI_orderedFromDend),0,102])

#Note: The x axis is not made of the columns names, but of their index: from 0 to 59. It identifies each (row#,col#) pair as a pixel
#If we want the x axis to be scaled with the proper values, we need to specify the extent of the data as extent=[xmin,xmax,ymin,ymax]
#Thus by having the first value be log(0.001) and the last be the maximum value of the array, we have the proper scaling on the x axis

fig3.suptitle('Hierarchical clustering and log(ISI) frequency matrix of each cell', fontsize = 24)

axISI_ordered.set_xticks([np.log(0.001), np.log(0.01), np.log(0.1), np.log(1)])
axISI_ordered.xaxis.set_ticklabels([0.001, 0.01, 0.1, 1], fontsize = 20)
axISI_ordered.set_xlabel('Inter Spike Interval (sec)', fontsize = 24)
axISI_ordered.set_ylabel('Cells', fontsize = 24)
axISI_ordered.set_title('Log(ISI) histogram for each cell\n as a grey scale density plot', fontsize = 24)

axISI_ordered.yaxis.set_ticks_position('none')
axISI_ordered.yaxis.set_ticklabels([])


axWard.set_xlabel('Euclidean distance', fontsize = 24)
axWard.set_title('Cluster tree of the cells based on the PCA\n of log(ISI) histograms and autocorrelograms', fontsize = 24)
axWard.tick_params(labelsize=20)

#Plot a vertical line at the 6ms mark on the figure
axISI_ordered.axvline(np.log(0.006),linestyle='--',color='red')
axISI_ordered.axhline(20, linestyle='--', color = 'blue')


#This will adjust the size of the plot automatically to be the size of the figure
axISI_ordered.set_aspect('auto')




























