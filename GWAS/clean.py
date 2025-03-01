# This was run locally, so I have included the code but with a placeholder filename since this will not be needed to run again. 


import pandas as pd
import numpy as np

filepath = "."

# In practice if we try to run this script we can use the actual filepath.

df = pd.read_csv("filepath") 

columns_to_keep = ["risk Allele", "pValue", "riskFrequency",  "or Value", "beta", "ci", "locations"]

df = df[columns_to_keep]

df["risk Allele"] = df["risk Allele"].astype(str).str[-1]
df["beta"] = np.log(df["or Value"])

print(df.head())

df.to_excel("cleaned_gwas.xlsx", index=False)