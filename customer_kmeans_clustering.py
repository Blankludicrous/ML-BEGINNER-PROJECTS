import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
#load dataset
df=pd.read_csv("customer.csv")
#define X
X=df[["Age","Annual_Income","Spending_Score","Online_Visits_Per_Month"]]
#define pipeline
pipeline=Pipeline(
    [
    ("imputation",SimpleImputer(strategy="mean")),
    ("scaling",StandardScaler()),
    ("model",KMeans())
    ]
)
#define silhouette score function
def silhouette_scorer(estimator,X):
    labels=estimator.fit_predict(X)
    return silhouette_score(X,labels)
#define params_grid
params_grid={
    "model__n_clusters":[2,3,4,5,6],
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
best_pipeline = grid.best_estimator_
clusters = best_pipeline.fit_predict(X)
#evaluate
print("inertia",best_pipeline.named_steps["model"].inertia_)
print("silhouette_score",silhouette_score(X,clusters))
print("best_paramters",grid.best_params_)
print("best_score",grid.best_score_)