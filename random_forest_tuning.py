import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
#load dataset
df=pd.read_csv("student_marks_new.csv")
#clear nan columns from label
df=df.dropna(subset=["pass"])
#load x and y
X=df[["study_hours","attendance","sleep_hours","previous_score","noise_feature"]]
y=df["pass"]
#train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#define pipeline
pipeline=Pipeline(
    [
        ("imputation",SimpleImputer()),
        ("rf",RandomForestClassifier(random_state=42,n_jobs=-1))
    ]
)
#loop for depths
depths=[None,5,10,15]
for d in depths:
    pipeline.set_params(
        rf__max_depth=d
    )
    pipeline.fit(X_train,y_train)
    train_acc=pipeline.score(X_train,y_train)
    test_acc=pipeline.score(X_test,y_test)
    print(f"tree depth={d} | train={train_acc} | test={test_acc}")
d=10
#loop for n_estimators
#d=best_depth
n_estimates=[50,100,200,300]
for n in n_estimates:
    pipeline.set_params(
        rf__max_depth=d,
        rf__n_estimators=n
    )
    pipeline.fit(X_train,y_train)
    train_acc=pipeline.score(X_train,y_train)
    test_acc=pipeline.score(X_test,y_test)
    print(f"n={n} | train={train_acc} | test={test_acc}")
n=50
#loop_for_leafs
#n=best_n
#d=best_depth
leafs=[1,2,5,10]
for leaf in leafs:
    pipeline.set_params(
        rf__max_depth=d,
        rf__n_estimators=n,
        rf__min_samples_leaf=leaf
    )
    pipeline.fit(X_train,y_train)
    train_acc=pipeline.score(X_train,y_train)
    test_acc=pipeline.score(X_test,y_test)
    print(f"leaf={leaf} | train={train_acc} | test={test_acc}")
leaf=5
mfs=["sqrt","log2",None]
for mf in mfs:
    pipeline.set_params(
        rf__max_depth=d,
        rf__n_estimators=n,
        rf__min_samples_leaf=leaf,
        rf__max_features=mf
    )
    pipeline.fit(X_train,y_train)
    train_acc=pipeline.score(X_train,y_train)
    test_acc=pipeline.score(X_test,y_test)
    print(f"mf-{mf} | train={train_acc} | test={test_acc}")
