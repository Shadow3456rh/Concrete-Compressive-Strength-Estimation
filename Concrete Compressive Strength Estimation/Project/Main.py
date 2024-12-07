import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


df = pd.read_excel("Dataset2.xlsx")
df.columns = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 
              'Coarse Aggregate', 'Fine Aggregate', 'Age', 'Compressive Strength']


x = df.drop('Compressive Strength', axis=1)
y = df['Compressive Strength']


model = GradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=5,
    max_features='log2',
    min_samples_leaf=4,
    min_samples_split=10,
    n_estimators=500,
    subsample=0.8,
    random_state=42
)

rf_model = RandomForestRegressor(
    max_depth=None,
    max_features=None,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=300,
    random_state=42
)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print(f'Gradient Mean Squared Error: {test_mse:.2f}')
print(f'Gradient R2 Score: {test_r2:.2f}')

rf_model.fit(x_train_scaled, y_train)
rf_pred = rf_model.predict(x_test_scaled)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print(f'Random mse: {rf_mse}')
print(f'Random r2= {rf_r2}')

kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = []
r2_scores = []

kfoldscaler = StandardScaler()
X_train_scaled = kfoldscaler.fit_transform(x)
for train_index, test_index in kf.split(X_train_scaled):
    X_train, X_test = X_train_scaled[train_index], X_train_scaled[test_index]
    Y_train, Y_test = y[train_index], y[test_index]
    
    
    model.fit(X_train, Y_train)
    
    
    Y_pred = model.predict(X_test)
    
    
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    
    mse_scores.append(mse)
    r2_scores.append(r2)

print(f"K-Fold Cross-Validation Results:")
print(f"MSE Scores: {mse_scores}")
print(f"Average MSE: {np.mean(mse_scores):.2f}")
print(f"R² Scores: {r2_scores}")
print(f"Average R²: {np.mean(r2_scores):.2f}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.7, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='-', label='Perfect Fit')
plt.xlabel('Actual Compressive Strength')
plt.ylabel('Predicted Compressive Strength')
plt.title('Actual vs Predicted Compressive Strength')
plt.legend()
plt.grid(True)
plt.show()

joblib.dump(model, 'concrete_strength_model.pkl')
joblib.dump(scaler, "scaler.pkl")

