'EOF'
import sys, pathlib
import pandas, numpy, sklearn, xgboost, matplotlib, seaborn

print("python     ", sys.version.split()[0])
for m in (pandas, numpy, sklearn, xgboost, matplotlib, seaborn):
    print(f"{m.__name__:<11}", m.__version__)

for f in sorted(pathlib.Path("data/raw").glob("*.csv")):
    print(f"{f.name:<24}{f.stat().st_size / 1e6:6.1f} MB")

# Smoke test: actually train with XGBoost so OpenMP is exercised, not just loaded
X = numpy.random.rand(500, 4); y = (X[:, 0] > 0.5).astype(int)
clf = xgboost.XGBClassifier(n_estimators=5, n_jobs=-1).fit(X, y)
print("xgboost train OK, accuracy:", round(clf.score(X, y), 3))
EOF