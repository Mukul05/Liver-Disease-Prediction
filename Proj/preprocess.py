import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import scatter_matrix
import csv


import types
import pandas as pd

def process(path):
    headings = ['Age','Gender','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio','Dataset']
    data=pd.read_csv(path,names = headings)
    print(data.head(n=5))
    
    data1 = data[data['Dataset']==2]
    data1 = data1.iloc[:,:-1]
    data2 = data[data['Dataset']==1]
    data2 = data2.iloc[:,:-1]
    
    fig = plt.figure(figsize=(10,15))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212,sharex=ax1)
    
    ax1.grid()
    ax2.grid()
    
    ax1.set_title('Features vs mean values',fontsize=13,weight='bold')
    ax1.text(200,0.8,'NO DISEASE',fontsize=20,horizontalalignment='center',color='green',weight='bold')
    
    ax2.set_title('Features vs mean values',fontsize=13,weight='bold')
    ax2.text(200,0.8,'DISEASE',fontsize=20,horizontalalignment='center',color='red',weight='bold')
    
    plt.sca(ax1)
    plt.xticks(rotation = 0,weight='bold',family='monospace',size='large')
    plt.yticks( weight='bold',family='monospace',size='large')
    
    plt.sca(ax2)
    plt.xticks(rotation = 0,weight='bold',family='monospace',size='large')
    plt.yticks( weight='bold',family='monospace',size='large')
    # sns.set_style('whitegrid')
    sns.barplot(data=data1,ax=ax1,orient='horizontal', palette='deep') # no disease
    sns.barplot(data=data2,ax=ax2,orient='horizontal',palette='deep',saturation=0.80) # with disease
    fig.savefig('results/Visualization.png')
    plt.pause(10)
    plt.show(block=False)
    plt.close()
    
    #Visualizing the differences in chemicals in Healthy/Unhealthy people
    with_disease = data[data['Dataset']==1]
    with_disease = with_disease.drop(columns=['Gender','Age','Dataset'])
    names1 = with_disease.columns.unique()
    mean_of_features1 = with_disease.mean(axis=0,skipna=True)
    without_disease = data[data['Dataset']==2]
    without_disease = without_disease.drop(columns=['Gender','Age','Dataset'])
    names2 = without_disease.columns.unique()
    mean_of_features2 = without_disease.mean(axis=0,skipna=True)
    people = []
    for x,y in zip(names1,mean_of_features1):
        people.append([x,y,'Diseased'])
    for x,y in zip(names2,mean_of_features2):
        people.append([x,y,'Healthy'])
    new_data = pd.DataFrame(people,columns=['Chemicals','Mean_Values','Status'])
    fig = plt.figure(figsize=(20,8))
    plt.title('Comparison- Diseased vs Healthy',size=20,loc='center')
    plt.xticks(rotation = 30,weight='bold',family='monospace',size='large')
    plt.yticks( weight='bold',family='monospace',size='large')
    
    g1 = sns.barplot(x='Chemicals',y='Mean_Values',hue='Status',data=new_data,palette="Blues_d")
    plt.legend(prop={'size': 20})
    plt.xlabel('Chemicals',size=19)
    plt.ylabel('Mean_Values',size=19)
    
    new_data
    fig.savefig('results/CHemicAL.png')
    plt.pause(10)
    plt.show(block=False)
    plt.close()
    
    #No of males and females
    fig= plt.figure(figsize=(15,6),frameon=False) 
    plt.title("Total Data",loc='center',weight=10,size=15)
    plt.xticks([]) # to disable xticks
    plt.yticks([]) # to disable yticks
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    only_gender = data['Gender']
    
    male_tot = only_gender[only_gender==0]
    no_of_male = len(male_tot)
    no_of_female = len(data) - len(male_tot)
    m_vs_f = [no_of_male,no_of_female]
    with_disease = data[data['Dataset']==1]
    not_with_disease = data[data['Dataset']==2]
    with_disease = with_disease['Gender']
    no_of_diseased = len(with_disease)
    no_of_not_diseased = len(data) - len(with_disease)
    
    d_vs_healthy = [no_of_diseased,no_of_not_diseased]
    ax1.axis('equal')
    ax2.axis('equal')
    # pie plot
    wedges, texts, autotexts= ax1.pie(m_vs_f,labels=('Male','Female'),radius=1,textprops=dict(color='k'),colors=['xkcd:ocean blue','xkcd:dark pink'],autopct="%1.1f%%")
    # pie plot
    wedges2, texts2, autotexts2 = ax2.pie(d_vs_healthy,labels=('Diseased','Not Diseased'),radius=1,textprops=dict(color='k'),colors=['#d95f02','#1b9e77'],autopct="%1.1f%%")
    plt.setp(autotexts,size=20)
    plt.setp(texts,size=20)
    plt.setp(autotexts2,size=20)
    plt.setp(texts2,size=20)
    fig.savefig('results/MalevsFemale.png')
    plt.pause(10)
    plt.show(block=False)
    plt.close()
    
    #Male vs Female statistics
    fig= plt.figure(figsize=(15,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    with_disease = data[data['Dataset']==1]
    not_with_disease = data[data['Dataset']==2]
    
    with_disease_m = with_disease[with_disease['Gender']==0]
    with_disease_m = with_disease['Gender']
    not_with_disease_m = not_with_disease[not_with_disease['Gender']==0]
    not_with_disease_m = not_with_disease['Gender']
    
    with_disease_f = with_disease[with_disease['Gender']==1]
    not_with_disease_f = not_with_disease[not_with_disease['Gender']==1]
    
    no_of_diseased_m = len(with_disease_m)
    no_of_not_diseased_m = len(not_with_disease_m)
    
    no_of_diseased_f = len(with_disease_f)
    no_of_not_diseased_f = len(not_with_disease_f)
    
    d_vs_healthy_m = [no_of_diseased_m, no_of_not_diseased_m]
    d_vs_healthy_f = [no_of_diseased_f, no_of_not_diseased_f]
    
    ax1.axis('equal')
    ax2.axis('equal')
    # pie plot
    
    wedges, texts, autotexts = ax1.pie(d_vs_healthy_m,labels=('Diseased','Not Diseased'),radius=1,textprops=dict(color='k'),colors=['#f46d43','#4575b4'],autopct="%1.1f%%")
    
    wedges2, texts2, autotexts2 = ax2.pie(d_vs_healthy_f,labels=('Diseased','Not Diseased'),radius=1,textprops=dict(color='k'),colors=['#f46d43','#4575b4'],autopct="%1.1f%%")
    
    plt.setp(autotexts,size=20)
    plt.setp(texts,size=20)
    
    plt.setp(autotexts2,size=20)
    plt.setp(texts2,size=20)
    
    ax1.text(0,0.04,'Male',size=20,color='#f7fcfd',horizontalalignment='center',weight='bold')
    ax2.text(0,0.04,'Female',size=20,color='#f7fcfd',horizontalalignment='center',weight='bold')
    
    fig.savefig('results/MVFStatistics.png')
    plt.pause(10)
    plt.show(block=False)
    plt.close()
    
    #Machine Learning We need to separate the target values from the rest of the table
    X = data.iloc[:,:-1].values
    t = data.iloc[:,-1].values
    #Gender column has entries as Male and Female. For a mathematical model to learn, we have to encode these into numbers.
    from sklearn.preprocessing import LabelEncoder
    lbl = LabelEncoder()
    X[:,1] = lbl.fit_transform(X[:,1])
    
    #Fill the missing rows with values
    data.isnull().any()
    data['Albumin_and_Globulin_Ratio'].isnull().sum()
    missing_values_rows = data[data.isnull().any(axis=1)]
    print(missing_values_rows)
    data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].mode().iloc[0])
    data['Albumin_and_Globulin_Ratio'].unique()
    data['Albumin_and_Globulin_Ratio'].value_counts()
    print(data['Albumin_and_Globulin_Ratio'].median())
    print(data['Albumin_and_Globulin_Ratio'].mean())
    data['Albumin_and_Globulin_Ratio'].isnull().sum()
    
    #Graphes
    print(data.head())    
    names=list(data.columns)
    correlations = data.corr()
    fig = plt.figure()
    fig.canvas.set_window_title('Correlation Matrix')
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,9,1)
    ticks = np.arange(0,9,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    fig.savefig('results/CorrelationMatrix.png')
    plt.pause(10)
    plt.show(block=False)
    plt.close()
    
    #scatterplot
    scatter_matrix(data)
    fig.savefig('results/ScatterMatrix.png')
    plt.pause(10)
    plt.show(block=False)
    plt.close()

    ncols=3
    plt.clf()
    f = plt.figure(1)
    f.suptitle(" Data Histograms", fontsize=12)
    vlist = list(data.columns)
    nrows = len(vlist) // ncols
    if len(vlist) % ncols > 0:
        nrows += 1
    for i, var in enumerate(vlist):
        plt.subplot(nrows, ncols, i+1)
        plt.hist(data[var].values, bins=15)
        plt.title(var, fontsize=10)
        plt.tick_params(labelbottom='off', labelleft='off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    fig.savefig('results/DataHistograms.png') 
    plt.pause(5)
    plt.show(block=False)
    plt.close()
    
    
    
    