import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
#load the dataset
df=pd.read_csv("students.csv")
#load X features
X=df[["Study_Hours_Per_Day","Sleep_Hours","Attendance_Percentage","Exam_Score"]]
#define pipeline
pipeline=Pipeline(
    [
        ("imputation",SimpleImputer(strategy="mean",add_indicator=True)),
        ("scaling",StandardScaler()),
        ("model",KMeans(random_state=42))
    ]
)
#define silhouette function
def silhouette_scorer(estimator,X):
    labels=estimator.fit_predict(X)
    return silhouette_score(X,labels)
#define params_grid
params_grid={
    "model__n_clusters":[2,3,4,5],
    "model__n_init":[10,20],
    "model__init":["k-means++","random"]
}
#define grid
grid=GridSearchCV(
    estimator=pipeline,
    param_grid=params_grid,
    cv=5,
    scoring=silhouette_scorer
)
grid.fit(X)
best_pipeline=grid.best_estimator_
clusters=best_pipeline.fit_predict(X)
print("inertia",best_pipeline.named_steps["model"].inertia_)
print("silhoutte_score",silhouette_score(X,clusters))
print("best_parameters",grid.best_params_)
print("best_Score",grid.best_score_)
#visulizing the clusters
X_transformed=best_pipeline[:-1].transform(X)
#define pca
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X)
#numerical cluster interpretation
centers=best_pipeline.named_steps["model"].cluster_centers_
center_orginals=best_pipeline[:-1].inverse_transform(centers)
centers_df=pd.DataFrame(
    center_orginals,
    columns=X.columns
)
print(centers_df)
#print the graph
plt.scatter(X_pca[:,0],X_pca[:,1],c=clusters)
plt.xlabel("component1")
plt.ylabel("component2")
plt.show()