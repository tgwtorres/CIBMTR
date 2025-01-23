import pandas as pd
import urllib.request as ur
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns

from kneed import KneeLocator
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
import missingno as msno


def plot_missing_values(df):
    msno.matrix(df)
    plt.title("Missing Values Matrix", fontsize=20)
    plt.show()


def cor_heatmap(df):
    corr_mat = df.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_mat, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()
 




def boxplot(df):
    df.reset_index(drop=True, inplace=True)
    plt.figure(figsize=(12, 12))
    df.boxplot(vert = False)
    plt.title('Boxplot of All Attributes')
    plt.xlabel('Values')
    plt.ylabel('Attributes')
    plt.show()
    
def plot_elbow_graph(sse, k_values, elbow_point):
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, sse, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.title('The Elbow Method showing the optimal k')
    

    plt.show()

def plot_silhouette_scores(X, sil_scores,range_n_clusters):
    silhouette_scores = sil_scores
    
    plt.figure(figsize=(10, 6))
    plt.plot(range_n_clusters, silhouette_scores, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Scores for Various Numbers of Clusters')
    plt.grid(True)
    plt.show()
    
def silhouette_plot (df,low,up):
    kmeans_kwargs = { "init": "random",
                 "n_init": 1,
                 "max_iter": 300,
                 "random_state": 0,
                }
    
    for k in [low, up]:
        plt.figure(figsize=(10,10))
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
        visualizer.fit(df)
        visualizer.show();