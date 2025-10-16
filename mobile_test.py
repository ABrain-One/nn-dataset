from ab.nn.api import mobile_data
import pandas as pd

# Fetch first 10 rows from `mobile`
df = mobile_data(max_rows=10)

print("Total rows available:", len(mobile_data()))
print("Showing first 10 rows:")
print(df.to_string(index=False))
