import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
#load dataset
df=pd.read_csv("fitness_kmeans.csv")
#define X
X=df[["Daily_Steps","Workout_Minutes","Calories_Burned","Active_Hours"]]
#define pipeline
pipeline=Pipeline([
    ("imputation",SimpleImputer(strategy="mean")),
    ("scaling",StandardScaler()),
    ("model",KMeans(random_state=42))
]
)
#elbow method
inertia_values=[]
k_range=range(1,10)
for k in k_range:
    temp_pipeline=Pipeline([
        ("imputation",SimpleImputer(strategy="mean")),
        ("scaling",StandardScaler()),
        ("model",KMeans(random_state=42,n_clusters=k))
    ])
    temp_pipeline.fit(X)
    inertia_values.append(
        temp_pipeline.named_steps["model"].inertia_
    )
plt.scatter(k_range,inertia_values)
plt.xlabel("number of clusters k")
plt.ylabel("inertia values")
plt.show()
#define silhouette score
def silhouette_scorer(estimator,X):
    labels=estimator.fit_predict(X)
    return silhouette_score(X,labels)
#define grid params
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
#fit and predict
grid.fit(X)
best_pipeline=grid.best_estimator_
clusters=best_pipeline.fit_predict(X)
print("inertia",best_pipeline.named_steps["model"].inertia_)
print("best paramets",grid.best_params_)
print("best score",grid.best_score_)
#visualization
X_transformed=best_pipeline[:-1].transform(X)
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_transformed)
#visualize output
plt.scatter(X_pca[:,0],X_pca[:,1],c=clusters)
plt.xlabel("componet1")
plt.ylabel("component2")
plt.show()
#interpretation
centers=best_pipeline.named_steps["model"].cluster_centers_
original_centers=best_pipeline[:-1].inverse_transform(centers)
centers_df=pd.DataFrame(
    original_centers,
    columns=X.columns
)
print(centers_df)