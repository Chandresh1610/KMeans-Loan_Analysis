import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle

df = pd.read_csv(r"C:\Users\Acer\OneDrive\Desktop\loan management.csv",index_col=0,header=0)

#print(df.shape)
#print(df.info())
#print(df.isnull().sum())

X = df.values[:,[1,2]]



from sklearn.cluster import KMeans
wsse = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i , random_state = 10)
    kmeans.fit(X)
    wsse.append(kmeans.inertia_)
#plt.plot(range(1,11),wsse)
#plt.scatter(range(1,11),wsse)
#plt.title('The Elbow method')
#plt.xlabel('No of clusters')
#plt.ylabel('WSSE')
#plt.show()
kmeans = KMeans(n_clusters =5,random_state = 10)
model = kmeans.fit(X)


df["Clusters"]=model

sns.lmplot( data = df,x = "BALANCE" , y="Loan cleared Percentage",
            fit_reg=False,
            hue='Clusters',palette="Set1")

df["Clusters"]=df.Clusters.map({0:"Cannot be given",1:"Can be given",2:"Must be given",3:"Not at all to be given",4:"Can be given but not now"})
new_df=df[df["Clusters"]=="Must be given"]

pickle.dump(model,open('model3.pkl','wb'))
model2 = pickle.load(open('model3.pkl','rb'))
