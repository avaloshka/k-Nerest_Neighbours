import pandas as pd
import numpy as np

url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
)

abalone = pd.read_csv(url)
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shoucked weight', 'Viscera weight', 'Shell weight', 'Rings']
abalone = abalone.drop('Sex', axis=1)
correction_matrix = abalone.corr()

# print(correction_matrix['Rings'])

# We will try to predict "Rings" so lets drop the column to train a set
X = abalone.drop("Rings", axis=1)
# Convert DataFrame to numpy array
X = X.values

# will use "Rings" as a target value
y = abalone["Rings"]
y = y.values

# We have a candidate for predictions, his parameters where taken and I will post his parameters in new_data_point
# They are in same order as original abalone df's columns
new_data_point = np.array([
    0.569552,
    0.446407,
    0.154437,
    1.016849,
    0.439051,
    0.222526,
    0.291208,
])

distances = np.linalg.norm(X- new_data_point, axis=1)

# find 3 nearest samples in database that are similar to new entry, get their id's
k = 3
nearest_neighbor_ids = distances.argsort()[:k]
# printed it would give you id's: array([4045, 1902, 1644])
# printed X[4045] will return values of 1st nearest neighbour

nearest_neighbor_rings = y[nearest_neighbor_ids]
# result in this case is ([9, 11, 10])

prediction = nearest_neighbor_rings.mean()
print(prediction)

