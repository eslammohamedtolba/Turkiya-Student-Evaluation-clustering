# Turkiya-Student-Evaluation-clustering
This project focuses on the analysis of student evaluations in Turkey. 
The code is designed to explore and analyze a dataset of student evaluations using various data analysis and clustering techniques.

## Prerequisites
Before running the code, make sure you have the following dependencies installed:
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- NumPy
- Scipy

## Overview of the Code
1-Loading the Dataset: The code starts by loading the student evaluation dataset named 'turkiye-student-evaluation_generic.csv' using Pandas.

2-Data Exploration: Various aspects of the dataset are explored:
- Displaying the first and last 5 rows of the dataset.
- Checking the shape of the dataset.
- Displaying statistical information about the dataset.

3-Data Visualization:
- Visualizing count plots for columns like 'instr,' 'class,' 'nb.repeat,' 'attendance,' and 'difficulty' to understand the distribution of values in these columns.

4-Data Cleaning:
- Checking for missing values in the dataset to decide whether data cleaning is needed. The code ensures that there are no missing values.

5-Exploring Question Data:
- Calculating the mean of the questions asked in the evaluation dataset.
- Displaying the mean values and creating a bar plot to visualize it.

6-Correlation Analysis:
- Calculating the correlation between features in the dataset.
- Displaying the correlation matrix as a heatmap to visualize relationships between variables.

7-Principal Component Analysis (PCA):
- Applying PCA to reduce the dimensionality of the dataset to 2 components.
- Calculating the amount of information retained using PCA.

8-Clustering Analysis:
- Using the K-Means clustering algorithm to cluster the data into three clusters based on PCA components.
- Visualizing the clusters and their centers.
- Counting the number of samples in each cluster.

9-Comparison with Full Question Data:
- Applying K-Means clustering to the full set of question data to compare the results with PCA-based clustering.
- Counting the number of samples in each cluster for the full question data.

10-Dendrogram and Agglomerative Clustering:
- Creating a dendrogram to visualize hierarchical clustering.
- Applying Agglomerative Clustering with two clusters and plotting the results.


## Cost Value
The code provided does not include any calculation or mention of a "cost value" of 38761.86.

## Contributions
Contributions to this project are welcome. If you have suggestions for improving the analysis, additional data to include, or other ideas to enhance the project, please feel free to contribute.

