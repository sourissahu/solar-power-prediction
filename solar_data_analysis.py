
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

"""**Importing dataset**"""

dts = pd.read_csv('/content/spg.csv')
X = dts.iloc[:, :-1].values
y = dts.iloc[:, -1].values

# prompt: print X as table

print(pd.DataFrame(X).to_string())

# prompt: print X as table

print(pd.DataFrame(y).to_string())

# y = np.reshape(y, (-1, 1))
# y

# Code for First Graph: Distribution of Generated Power
plt.figure(figsize=(10, 6))
sns.histplot(data['generated_power_kw'], bins=30, kde=True)
plt.title("Distribution of Generated Power (kW)")
plt.xlabel("Generated Power (kW)")
plt.ylabel("Frequency")
plt.show()

# Scatter Plots for Key Variables vs. Generated Power
# Create a grid of scatter plots for selected features vs. generated_power_kw
key_features = ['temperature_2_m_above_gnd', 'total_cloud_cover_sfc', 'shortwave_radiation_backwards_sfc',
                'wind_speed_10_m_above_gnd', 'angle_of_incidence']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, feature in enumerate(key_features):
    sns.scatterplot(x=data[feature], y=data['generated_power_kw'], ax=axes[i // 3, i % 3])
    axes[i // 3, i % 3].set_title(f"{feature} vs. Generated Power")
    axes[i // 3, i % 3].set_xlabel(feature)
    axes[i // 3, i % 3].set_ylabel("Generated Power (kW)")
plt.tight_layout()
plt.show()

data = pd.read_csv('spg.csv')
corr = data.corr()
plt.figure(figsize=(22,22))
sns.heatmap(corr, annot=True, square=True);

# ## Splitting Training and Test Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("Train Shape: {} {} \nTest Shape: {} {}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

X_train

X_test

sc_X = StandardScaler()
X_train_transformed = sc_X.fit_transform(X_train)
X_test_transformed = sc_X.transform(X_test)

sc_y = StandardScaler()
y_train_transformed = sc_y.fit_transform(y_train.reshape(-1, 1))
y_test_transformed = sc_y.transform(y_test.reshape(-1, 1))

X_test_transformed

aa=sc_X.inverse_transform(X_test_transformed)
aa

"""**Creating the Neural Network with Scikit-learn's MLPRegressor**"""

# Define MLP model
mlp = MLPRegressor(hidden_layer_sizes=(32, 64), activation='relu', solver='adam',
                   max_iter=150, random_state=0)
mlp.fit(X_train_transformed, y_train_transformed.ravel())  # Flatten y_train for scikit-learn compatibility

# Display a summary of the network configuration
print(f"MLPRegressor configuration:\nLayers: {mlp.hidden_layer_sizes}\nActivation: {mlp.activation}\nSolver: {mlp.solver}")

!pip install graphviz networkx

import matplotlib.pyplot as plt
import networkx as nx

def plot_mlp_structure(layers, save_as='mlp_structure.png'):
    # Initialize a directed graph
    G = nx.DiGraph()

    # Define node positions for input, hidden, and output layers
    layer_sizes = [X_train.shape[1]] + list(layers) + [1]  # Includes input and output layer sizes
    pos = {}
    node_idx = 0

    for layer_num, layer_size in enumerate(layer_sizes):
        for i in range(layer_size):
            pos[node_idx] = (layer_num, i - layer_size / 2)
            node_idx += 1

    # Add edges between layers to represent weights/connections
    node_idx = 0
    for layer_num, layer_size in enumerate(layer_sizes[:-1]):
        next_layer_size = layer_sizes[layer_num + 1]
        for i in range(layer_size):
            for j in range(next_layer_size):
                G.add_edge(node_idx + i, node_idx + layer_size + j)
        node_idx += layer_size

    # Draw the network
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=False, node_size=100, node_color="skyblue", edge_color="gray")
    plt.title("MLP Network Structure")
    plt.savefig(save_as)
    plt.show()

# Example usage with layers [32, 64] (input layer is inferred from the dataset)
plot_mlp_structure([32, 64], save_as='mlp_structure.png')

"""**Evaluation and Prediction**"""

y_train_transformed

y_train_orig = sc_y.inverse_transform(y_train_transformed)
y_train_orig  # Original training target values

y_test_transformed

y_test_orig = sc_y.inverse_transform(y_test_transformed)
y_test_orig # Original test target values

aa=mlp.predict(X_train_transformed)
aa

y_pred_train = sc_y.inverse_transform(aa.reshape(-1, 1)) # Predicted training values
y_pred_train

bb=mlp.predict(X_test_transformed)
print(bb)
y_pred_test = sc_y.inverse_transform(bb.reshape(-1, 1))    # Predicted test values

y_pred_test

"""**Calculate Performance Metrics**

"""

train_rmse = mean_squared_error(y_train_orig, y_pred_train, squared=False)
test_rmse = mean_squared_error(y_test_orig, y_pred_test, squared=False)
train_r2 = r2_score(y_train_orig, y_pred_train)
test_r2 = r2_score(y_test_orig, y_pred_test)

print("Training RMSE:", train_rmse)
print("Testing RMSE:", test_rmse)
print("Training R^2:", train_r2)
print("Testing R^2:", test_r2)

plt.figure(figsize=(16, 12))

# Training Predictions vs Actual
plt.subplot(2, 2, 1)
plt.scatter(y_train_orig, y_pred_train, color="blue", label="Train Data")
plt.plot([y_train_orig.min(), y_train_orig.max()], [y_train_orig.min(), y_train_orig.max()], 'k--', lw=2)
plt.xlabel('Actual Generated Power')
plt.ylabel('Predicted Generated Power')
plt.title('Training Set Predictions')
plt.legend()

# Test Predictions vs Actual
plt.subplot(2, 2, 2)
plt.scatter(y_test_orig, y_pred_test, color="red", label="Test Data")
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'k--', lw=2)
plt.xlabel('Actual Generated Power')
plt.ylabel('Predicted Generated Power')
plt.title('Test Set Predictions')
plt.legend()

# Training Data by Solar Azimuth
x_axis = sc_X.inverse_transform(X_train)[:,-1]
x2_axis = sc_X.inverse_transform(X_test)[:,-1]
plt.subplot(2, 2, 3)
plt.scatter(x_axis, y_train_orig, label='Actual Generated Power', color="blue")
plt.scatter(x_axis, y_pred_train, color='cyan', label='Predicted Generated Power')
plt.xlabel('Solar Azimuth')
plt.ylabel('Generated Power')
plt.title('Training Predictions vs Solar Azimuth')
plt.legend()

# Test Data by Solar Azimuth
plt.subplot(2, 2, 4)
plt.scatter(x2_axis, y_test_orig, label='Actual Generated Power', color="red")
plt.scatter(x2_axis, y_pred_test, color='orange', label='Predicted Generated Power')
plt.xlabel('Solar Azimuth')
plt.ylabel('Generated Power')
plt.title('Test Predictions vs Solar Azimuth')
plt.legend()

plt.tight_layout()
plt.show()

results = np.concatenate((y_test_orig, y_pred_test), 1)
results = pd.DataFrame(data=results)
results.columns = ['Real Solar Power Produced', 'Predicted Solar Power']
pd.options.display.float_format = "{:,.2f}".format
results[7:18]

sc = StandardScaler()
pred_whole = mlp.predict(sc.fit_transform(X))
pred_whole_orig = sc_y.inverse_transform(pred_whole.reshape(-1, 1))
pred_whole_orig

r2_score(pred_whole_orig, y)

y_pred_train

df_results = pd.DataFrame.from_dict({
    'R2 Score of Whole Data Frame': r2_score(pred_whole_orig, y),
    'R2 Score of Training Set': r2_score(y_pred_train, y_train_orig),
    'R2 Score of Test Set': r2_score(y_pred_test,y_test_orig),
    'Mean of Test Set': np.mean(y_pred_test),
    'Standard Deviation pf Test Set': np.std(y_pred_test),
    'Relative Standard Deviation': np.std(y_pred_test) / np.mean(y_pred_test),
},orient='index', columns=['Value'])
display(df_results.style.background_gradient(cmap='afmhot', axis=0))

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.001)
lasso.fit(X_train_transformed, y_train_transformed)
lasso_coeff = pd.DataFrame({'Feature Importance': lasso.coef_}, index=dts.columns[:-1])

# Plotting non-zero coefficients
lasso_coeff[lasso_coeff['Feature Importance'] != 0].sort_values('Feature Importance').plot(kind='barh', figsize=(6,6), cmap='winter')
plt.title("Lasso Feature Importance")
plt.show()
