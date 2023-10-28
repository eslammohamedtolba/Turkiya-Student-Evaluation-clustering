# Import required dependencies
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.cluster import KMeans # clustering
from collections import Counter


# Load dataset
evaluation_dataset = pd.read_csv('turkiye-student-evaluation_generic.csv')
# Show the first five rows
evaluation_dataset.head()
# Show the last five rows
evaluation_dataset.tail()
# Show dataset shape
evaluation_dataset.shape
# Show some statistical info about the dataset
evaluation_dataset.describe()


# count_values function takes the dataset and a column and return the count values in this column
def count_values(dataset,column_name):
    return (dataset[column_name].value_counts())

# count number of unique values in the instr, class, nb.repeat, attendance and difficulty columns and plot it
columns_names = ['instr','class','nb.repeat','attendance','difficulty']
for column_name in columns_names:
    plt.figure(figsize=(5,5))
    sns.countplot(x = column_name,data = evaluation_dataset)
    plt.show()
    print(f"the count values of {column_name} column are: ")
    print(count_values(evaluation_dataset,column_name),end="\n\n")


# Check about the none(missing) values in the dataset to decide if will make data cleaning or not
evaluation_dataset.isnull().sum().sum()


# Find mean of quesions
quesions = evaluation_dataset.iloc[:,5:33]
quesions_mean = quesions.mean()
total_mean = quesions_mean.mean()
print(total_mean)
quesions_mean.head()
# show the bar plot between mean and index columns in quesions_mean dataset
plt.figure(figsize=(10,10))
sns.barplot(x='index',y='mean',data=quesions_mean)
plt.show()


# Find correlation between featuers in the dataset
correlation_values = evaluation_dataset.corr()
# Draw the correlation
plt.figure(figsize=(15,15))
sns.heatmap(correlation_values,fmt='.1f',cbar=True,square=True,annot=True,annot_kws={'size':8},cmap="Blues")

quesions.head()
# Use principle component analysis to decomposite the 28 columns into 2 columns
from sklearn.decomposition import PCA
pca = PCA(n_components=2,random_state=42)
decomposed_quesions = pca.fit_transform(quesions)
print(decomposed_quesions)
# How much info we retained from the dataset
pca.explained_variance_ratio_.cumsum()[1]



# Create the model
distortions = []
for n in range(1,11):
    KModel = KMeans(n_clusters=n,init='k-means++')
    KModel.fit(decomposed_quesions)
    distortions.append(KModel.inertia_)
# elbow method
plt.figure(figsize=(7,7))
plt.plot(range(1,11),distortions,color='red',marker='x')
# Choose optimal number of clusters = 3 and train the model on
optimal_Num_clusters = 3
KModel = KMeans(n_clusters=optimal_Num_clusters,init='k-means++')
KModel.fit(decomposed_quesions)
KModel.inertia_
# labels of three clusters samples on pca quesions
labels = KModel.labels_
labels
centers = KModel.cluster_centers_
centers
# Plot clusters groups and its centers
plt.figure(figsize=(10,10))
plt.scatter(decomposed_quesions[labels==0,0],decomposed_quesions[labels==0,1],color='red',s=50,marker='x',label='cluster 1')
plt.scatter(decomposed_quesions[labels==1,0],decomposed_quesions[labels==1,1],color='blue',s=50,marker='x',label='cluster 2')
plt.scatter(decomposed_quesions[labels==2,0],decomposed_quesions[labels==2,1],color='green',s=50,marker='x',label='cluster 3')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('clusters and its centers')
plt.scatter(centers[:,0],centers[:,1],color='black',s=150,marker='o',label='centers')
plt.legend()
plt.show()
# count number of samples in each cluster
Counter(labels)

# Try the model on all quesions to compare the samples in each cluster with the samples in each cluster of the model of pca method
optimal_Num_clusters = 3
KModel_quesions = KMeans(n_clusters=optimal_Num_clusters,init='k-means++')
KModel_quesions.fit(quesions)
# label of all samples with all quesions
labels_quesions = KModel_quesions.labels_
# count number of samples in each cluster
Counter(labels_quesions)



# dendogram method
import scipy.cluster.hierarchy as hier
dendogram = hier.dendrogram(hier.linkage(decomposed_quesions,method='ward'))
plt.title('Dendogram')
plt.xlabel('quesions')
plt.ylabel('Distance')
plt.show()
# AgglomerativeClustering method on two clusters
from sklearn.cluster import AgglomerativeClustering
AggClusteringModel = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
label_Agg = AggClusteringModel.fit_predict(decomposed_quesions)
# Plot clusters groups
plt.figure(figsize=(10,10))
plt.scatter(decomposed_quesions[label_Agg==0,0],decomposed_quesions[label_Agg==0,1],color='red',s=50,marker='x',label='cluster 1')
plt.scatter(decomposed_quesions[label_Agg==1,0],decomposed_quesions[label_Agg==1,1],color='blue',s=50,marker='x',label='cluster 2')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('clusters')
plt.legend()
plt.show()
# count number of samples in each cluster
Counter(label_Agg)

