import sys, pathlib
import pandas, numpy, sklearn, xgboost, matplotlib, seaborn

print("python     ", sys.version.split()[0])
for m in (pandas, numpy, sklearn, xgboost, matplotlib, seaborn):
    print(f"{m.__name__:<11}", m.__version__)

for f in sorted(pathlib.Path("data/raw").glob("*.csv")):
    print(f"{f.name:<24}{f.stat().st_size / 1e6:6.1f} MB")