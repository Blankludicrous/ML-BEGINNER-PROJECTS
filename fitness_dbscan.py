import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
#load dataset
df=pd.read_csv("fitness.csv")
#define X
X=df[["Daily_Steps","Workout_Minutes","Calories_Burned","Active_Hours"]]
#define pipeline
pipeline=Pipeline([
    ("imputation",SimpleImputer(strategy="mean")),
    ("scaling",StandardScaler()),
    ("model",DBSCAN())
])
#define silhouette score
def silhouette_scorer(estimator,X):
    labels=estimator.fit_predict(X)
    masks=labels!=-1
    if len(set(labels[masks]))<2:
        return -1
    score=silhouette_score(X[masks],labels[masks])
    return score
#define grid params
params_grid={
    "model__eps":[0.3,0.5,0.7],
    "model__min_samples":[3,5,7]
}
#define grid
grid=GridSearchCV(
    estimator=pipeline,
    param_grid=params_grid,
    cv=5,
    scoring=silhouette_scorer
)
#fit and predict data
grid.fit(X)
best_pipeline=grid.best_estimator_
clusters=best_pipeline.fit_predict(X)
print("best_parameters",grid.best_params_)
print("best_Score",grid.best_score_)
#visualization
X_transformed=best_pipeline[:-1].transform(X)
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_transformed)
plt.scatter(X_pca[:,0],X_pca[:,1],c=clusters)
plt.xlabel("component 1")
plt.ylabel("component 2")
plt.title("dbscan")
plt.show()
#cluster mean
df["clusters"]=clusters
cluster_summary=df.groupby("clusters").mean()
print(cluster_summary)
#cluster sizes
unique,counts=np.unique(clusters,return_counts=True)
print(dict(zip(unique,counts)))
#cluster core samples
core_samples=best_pipeline.named_steps["model"].core_sample_indices_
print("Number of core points",len(core_samples))
#detect outliers
outliers=df["clusters"]==-1
print(outliers)