import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
import warnings
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set()

raw_data = pd.read_csv('C:\\Users\\Boris\\Desktop\\playground-series-s3e20\\train.csv')
df = raw_data.copy()

# Handle missing value
df_missing_values = pd.DataFrame(data=df.isnull().sum().sort_values(ascending=False), columns=['missing'])
df_missing_values.plot.bar(figsize=(15, 3), fontsize=7)
df_missing_values.plot(kind='bar', figsize=(15,3), fontsize=7)
# delete some NANs
columns_to_drop = df_missing_values.query("missing > 20000").index
df = df.drop(columns_to_drop, axis=1)
# fill NAN with mean value
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputation = imputer.fit_transform(df.drop('ID_LAT_LON_YEAR_WEEK',axis=1))
df2 = pd.DataFrame(data=imputation, columns=df.drop('ID_LAT_LON_YEAR_WEEK', axis=1).columns)
df2['ID_LAT_LON_YEAR_WEEK'] = df['ID_LAT_LON_YEAR_WEEK']
df2['year'] = df2['year'].astype('int32')
df2['week_no'] = df2['week_no'].astype('int32')

fig, axs = plt.subplots(15,5,figsize=(12,50))
fig.subplots_adjust(hspace=0.5)
fig.tight_layout()

axs = axs.flatten()

for i, column in enumerate(df.describe().columns):
    axs[i].hist(df[column])
    axs[i].set_title(column, size=6)

df2['year_week'] = pd.to_datetime(df2['year'].astype(str) + df['week_no'].astype(str) + '0', format='%G%V%w')
df2['LAT_LON'] = df2['latitude'].astype('string') + '_' + df2['longitude'].astype('string')

df2_loc_mean = df2.groupby('LAT_LON').agg('mean').sort_values('emission', ascending=False)
df2_loc_sum = df2.groupby('LAT_LON').agg('sum').sort_values('emission', ascending=False)

# create subplots structure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# assign plots to subplots
df2_loc_mean.emission.plot.line(ax=ax1)
df2_loc_sum.emission.plot.line(ax=ax2)

# style subplots for readability
ax1.tick_params(axis='x', labelrotation=45)
ax2.tick_params(axis='x', labelrotation=45)
ax1.set_title('average emissions by location')
ax2.set_title('total emissions by location')
ax1.set_ylabel('Emissions')

plt.show()

# create subplots
fig, ax = plt.subplots(figsize=(8, 6))

# get country from geopandas
countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
countries[countries["name"] == "Rwanda"].plot(color="lightgrey", ax=ax)

# plot data on top
df2.plot.scatter(x="longitude",
                 y="latitude",
                 s='emission',
                 c='emission',
                 ax=ax)
ax.grid(alpha=0.5)
plt.show()

# Show emissions on a better map for better visual representation

fig = go.Figure(go.Densitymapbox(lat=df2['latitude'],
                                 lon=df2['longitude'],
                                 z=df2['emission'],
                                 radius=5,
                                 colorscale='bluered'))

fig.update_layout(mapbox_style="carto-positron",
                  mapbox_center_lon=30.34,
                  mapbox_center_lat=-2.01,
                  mapbox_zoom=6)

fig.show()

# Understand how average emissions fluctuate over the time period
sns.lineplot(data=df2, x='year_week', y='emission', hue='year')

# Generate unique locations where emissions are unusually high
locations_above_600 = set(df2.query("emission > 600")['LAT_LON'])

# Look at the outlier locations in isolation
for location in locations_above_600:
    df2.query("LAT_LON == @location").plot.line(x='year_week', y='emission', title=location, figsize=(12, 4))
    plt.show()

# Drop non-numerical features
df3 = df2.drop(['ID_LAT_LON_YEAR_WEEK', 'emission', 'year_week'], axis=1)

# Look at correlation of variables
corrs = df3.corrwith(df2['emission']).sort_values(ascending=False)
corrs.plot.bar(figsize=(15, 3), fontsize=7)

# Top correlated features
corrs.head()

threshold = df2.emission.quantile(0.995)
df_no_outliers = df2.drop(['LAT_LON', 'year_week', 'ID_LAT_LON_YEAR_WEEK'], axis=1).query("emission < @threshold")

X = df_no_outliers.drop('emission', axis=1)
y = df_no_outliers['emission']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=12)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

def fit_estimator(model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()
    print(model)

    cpu_time = end_time - start_time
    print(f"> CPU Time: {cpu_time:.2f} seconds")

    mse = np.mean((y_test - y_pred) ** 2)
    print(f"> Mean Squared Error (MSE): {mse}")

    mae = np.mean(np.abs(y_test - y_pred))
    print(f"> Mean Absolute Error (MAE): {mae}")

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"> Root Mean Squared Error: {rmse}")

    score = model.score(X_test, y_test)
    print(f"> Score: {score}")


tree = DecisionTreeRegressor()
fit_estimator(tree, X_train, y_train, X_test, y_test)

line = LinearRegression()
fit_estimator(line, X_train, y_train, X_test, y_test)

forest = RandomForestRegressor()
fit_estimator(forest, X_train, y_train, X_test, y_test)

svr = SVR()
fit_estimator(svr, X_train, y_train, X_test, y_test)


