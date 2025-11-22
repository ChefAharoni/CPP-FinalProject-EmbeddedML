import pandas as pd
from sklearn.datasets import make_blobs

features, labels = make_blobs(
    n_samples=1000, 
    centers=3, 
    n_features=2, 
    random_state=42
)

df = pd.DataFrame(features, columns=['blob_g1', 'blob_g2'])
df['label'] = labels

df.to_csv('gaussian_blobs.csv', index=False)

print("Dataset generated and saved as 'gaussian_blobs.csv'")
print(df.head())