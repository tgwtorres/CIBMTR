import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from kneed import KneeLocator
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy import stats

def drop_attributes(df,threshold):
    cols = df.columns
    for col in cols:
        if df[col].isnull().sum() / len(df) > threshold:
            df.drop(col, axis=1, inplace=True)
            print(f"Column {col} dropped")
    return df

def missing_val_df(df):
    missingCount = df.isnull().sum()
    missing_percentage = (missingCount/ len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Missing Values': missingCount,
        'Missing Percent': missing_percentage,
        'Data Type': df.dtypes
    })

    missing_df.sort_values(by='Missing Percent', ascending=False, inplace=True)
    missing_df = missing_df.reset_index()
    missing_df = missing_df.rename(columns={'index': 'variables'})

    return missing_df

    

def normalize (df):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df)
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
    return normalized_df

def drop_duprows(df):
    duplicate_rows = df[df.duplicated(keep=False)] 
    df.drop_duplicates(keep='first', inplace=True)
        
    
def drop_row_missing_val (df,threshold):
    thresh = int((1 - threshold) * df.shape[1])
    original_rows = df.shape[0]
    # Drop rows with missing value count above the threshold
    df_dropped = df.dropna(thresh=thresh)
    # Calculate the number of rows removed
    new_rows = df_dropped.shape[0]
    print(f"Removed {original_rows - new_rows} rows with missing values")
    
    return df_dropped

def check_missing_values(df):
    # Returns a Pandas Series with counts of missing values in each column
    return df.isnull().sum()

def simple_impute(df,strategy='mean'):
    if strategy == 'constant':
        imputer = SimpleImputer(strategy=strategy, fill_value=0)  # You can change fill_value if needed
    else:
        imputer = SimpleImputer(strategy=strategy)
    
    # Fit on the dataset and transform it to impute missing values
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    
    return df_imputed

def merge (df1,df2):
    merged_data = pd.concat([df1,df2],axis =1)
    return merged_data

def feature_selection_varianceThreshold(df,th):
    sel = VarianceThreshold(threshold=(th * (1 - th)))
    sel.fit_transform(df)
    selected_indices = sel.get_support(indices=True)
    
    selected_feature_names = df.columns[selected_indices]
    return selected_feature_names

def correlated_att_handling(df,threshold):
    corr_matrix = df.corr().abs()
    # Create a mask to identify highly correlated features
    mask = (corr_matrix >= threshold) & (corr_matrix < 1.0)
    # Identify columns to drop
    columns_to_drop = set()
    for col in mask.columns:
        correlated_cols = mask.index[mask[col]].tolist() ##(Written By ChatGPT3.5)
        if correlated_cols:
            columns_to_drop.update(correlated_cols)
    # Remove highly correlated columns
    df_cleaned = df.drop(columns=columns_to_drop)
    return df_cleaned

def scaling(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    return scaled_df

def remove_outliers (df,threshold):
    outlier_indices = []
    
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            avg = df[col].mean()
            std = df[col].std()
            
            # Identify outliers based on the threshold
            lower_bound = avg - (threshold * std)
            upper_bound = avg + (threshold * std)
            
            # Get the indices of outliers in this column
            column_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            
            # Extend the outlier_indices list with the indices of outliers in this column
            outlier_indices.extend(column_outliers)
    
    # Get unique indices of removed outliers
    unique_outlier_indices = np.unique(outlier_indices)
    
    # Remove the outliers from the DataFrame
    cleaned_df = df.drop(index=unique_outlier_indices)
    
    return cleaned_df

def get_num_cols(df):
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    return num_cols


def numeric_df(df):
    numeric_cols = df.select_dtypes(include=['number'])
    return numeric_cols

def get_cat_cols(df):
    categorical_cols = df.select_dtypes(include = ['object','category']).columns.tolist()
    return categorical_cols

def categorical_df(df):
    categorical_cols = df.select_dtypes(include = ['object','category'])
    return categorical_cols

def output_file (filename,directory,df):
    fileN =  os.path.join(directory,filename)
    df.to_csv(fileN, index=True)

def remove_outliers_zscore(data, threshold):
    z_scores = np.abs(stats.zscore(data))
    filtered_data = data[(z_scores < threshold)]
    return filtered_data

    
def count_missing_vals (df):
    missing_val_count = df.isnull().sum()
    return missing_val_count

def data_split (X,y,random_state):
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def find_best_k(df, max_k):
    sse = []  
    k_values = range(1, min(len(df), max_k+1))  
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)  

    # Find the elbow point
    kl = KneeLocator(k_values, sse, curve='convex', direction='decreasing')
    
    return kl.elbow, sse
    
def compute_silhouette_coefficients(X, range_n_clusters,RndState):
    silhouette_scores = []
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    return silhouette_scores

def find_best_cluster_number(X, range_n_clusters,RndState):
    silhouette_scores = compute_silhouette_coefficients(X, range_n_clusters,RndState)
    best_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
    best_score = np.max(silhouette_scores)
    return best_n_clusters, best_score  

def dbscan (X):
    X_normalized = StandardScaler().fit_transform(X)

    db = DBSCAN(eps=0.5, min_samples=5).fit(X_normalized)
    cluster_labels = db.labels_

    n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise_ = list(cluster_labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    return n_clusters_

def kmeans_clustering(n_cluster, df,RndState):
    kmeans = KMeans(n_clusters=n_cluster, random_state=RndState)
    kmeans.fit(df)
    labels = kmeans.labels_
    
    df['cluster'] = labels
    
def inner_join(df1,df2,key):
    result = pd.merge(df1, df2, on=key, how='inner')
    return result

def binary_encoding(df, label, positive_label):
    mapping = {positive_label: 1}
    df[label] = df[label].map(lambda x: mapping.get(x, 0))
    return df

def perform_PCA(df,n_components):
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(df)
    
    pca_df = pd.DataFrame(
        pca_data,
        index=df.index,
        columns=[f'PC{i+1}' for i in range(pca_data.shape[1])]
    )
    # Step 3: Return the transformed data and the PCA object
    return pca_df, pca

def apply_PCA (test_data, pca_mod):
    scaled_test_data = test_data
    
    # Step 2: Apply the PCA transformation using the fitted PCA model
    pca_test_data = pca_mod.transform(scaled_test_data)
    
    # Step 3: Create a DataFrame with the PCA results for the test data
    pca_test_df = pd.DataFrame(
        pca_test_data,
        index=test_data.index,
        columns=[f'PC{i+1}' for i in range(pca_test_data.shape[1])]
    )
    
    return pca_test_df

def remove_negative_silhouette_samples(X, labels):
    silhouette_vals = silhouette_samples(X, labels)
    
    positive_mask = silhouette_vals >= 0
    
    df_positive = X[positive_mask].copy()
    df_positive['Cluster_Labels'] = labels[positive_mask]
    
    negative_indices = X.index[~positive_mask].tolist()
    
    return df_positive
