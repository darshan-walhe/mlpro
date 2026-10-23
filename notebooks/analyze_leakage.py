import pandas as pd
import numpy as np

from src.utils import prepare_dataset

X, y, df = prepare_dataset()

# Print correlation of numeric features with 'y'
X_num = X.select_dtypes(include=[np.number])
correlations = X_num.corrwith(y)
print("Numeric correlations with target:")
print(correlations.sort_values(ascending=False))

# Crosstab 'Has Special Considerations'
for col in ["Has Special Considerations", "Class of Travel", "Quota"]:
    print(f"\n--- {col} ---")
    print(pd.crosstab(df[col], y, normalize='index'))

# Are there any features completely separating the data?
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=1)
for col in X_num.columns:
    tree.fit(X_num[[col]], y)
    print(f"{col} tree score: {tree.score(X_num[[col]], y):.4f}")

