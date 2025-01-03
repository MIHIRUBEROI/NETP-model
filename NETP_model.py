import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load and filter data
all_point_path = '/Users/mihiruberoi/Documents/coursework/fall_2024/Tibet_Final/all_point.csv'
all_point_df = pd.read_csv(all_point_path)
filtered_all_point_df = all_point_df[all_point_df['culture'].isin(['Nuomuhong', 'Kayue'])]
filtered_all_point_df.to_csv('/Users/mihiruberoi/Documents/coursework/fall_2024/Tibet_Final/filtered_all_point.csv', index=False)

# Read necessary files
filtered_all_point_df = pd.read_csv("/Users/mihiruberoi/Documents/coursework/fall_2024/Tibet_Final/filtered_all_point.csv")
cultural_relics_df = pd.read_csv("/Users/mihiruberoi/Documents/coursework/fall_2024/Tibet_Final/cultural_relics.csv")
dated_sites_df = pd.read_csv("/Users/mihiruberoi/Documents/coursework/fall_2024/Tibet_Final/dated_sites.csv")

# Rename columns for consistency
dated_sites_df.rename(columns={
    'Site name': 'Site',
    'Longitude(E)': 'Longitude',
    'Latitude(N)': 'Latitude'
}, inplace=True)

# Merge dataframes to get latitude and longitude
df_cult = cultural_relics_df[['Site', 'Latitude', 'Longitude']]
df_dated = dated_sites_df[['Site', 'Latitude', 'Longitude']]
df_merged = pd.merge(df_cult, df_dated, how='outer', on='Site', suffixes=('_cult', '_dated'))
df_merged['latitude'] = df_merged['Latitude_cult'].fillna(df_merged['Latitude_dated'])
df_merged['longitude'] = df_merged['Longitude_cult'].fillna(df_merged['Longitude_dated'])
df_merged = df_merged[['Site', 'latitude', 'longitude']]

# Combine with filtered data
final_df = filtered_all_point_df.merge(
    df_merged,
    how='left',
    left_on='site_name',  
    right_on='Site'
)

# Clean up missing data
final_df.dropna(subset=['latitude', 'longitude'], inplace=True)
final_df.drop(columns='Site', inplace=True)

# Sample Kayue sites
site_counts = final_df['culture'].value_counts()
kayue_sites = final_df[final_df['culture'] == 'Kayue']
nuomuhong_sites = final_df[final_df['culture'] == 'Nuomuhong']

if len(kayue_sites) >= 45:
    kayue_sample = kayue_sites.sample(n=45, random_state=42)
else:
    kayue_sample = kayue_sites
    print(f"Only {len(kayue_sites)} Kayue sites available; using all of them.")

final_df = pd.concat([kayue_sample, nuomuhong_sites]).reset_index(drop=True)

# Clean columns
nuomuhong_df = final_df[final_df['culture'] == 'Nuomuhong']
kayue_df = final_df[final_df['culture'] == 'Kayue']

cols_to_drop = ['geometry', 'Unnamed: 3', 'fluctuation', 'temperature']
for df_temp in [nuomuhong_df, kayue_df]:
    for col in cols_to_drop:
        if col in df_temp.columns:
            df_temp.drop(columns=[col], inplace=True, errors='ignore')

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Nuomuhong Sites DataFrame (Updated):")
    print(nuomuhong_df)
    print("\nKayue Sites DataFrame (Updated):")
    print(kayue_df)

df_filtered = pd.concat([nuomuhong_df, kayue_df], ignore_index=True)

# Process soil erosion data
if 'soil_erosi' in df_filtered.columns:
    df_filtered['soil_erosi'] = df_filtered['soil_erosi'].astype(str)
    df_filtered['soil_erosi_no_decimal'] = df_filtered['soil_erosi'].apply(lambda x: x.split('.')[0])
    
    def split_erosion_code(code_str):
        code_str = code_str.strip()
        if len(code_str) == 1:
            return (code_str[0], "0")
        else:
            return (code_str[0], code_str[1:])
    
    df_filtered[['erosion_main', 'erosion_strength']] = df_filtered['soil_erosi_no_decimal'] \
        .apply(lambda s: pd.Series(split_erosion_code(s)))
    
    df_filtered['erosion_strength'] = df_filtered['erosion_strength'].astype(int)
    
    erosion_mapping = {
        "1": "hydraulic_erosion",
        "2": "wind_erosion",
        "3": "freeze-thaw_erosion"
    }
    df_filtered['erosion_main'] = df_filtered['erosion_main'].map(erosion_mapping)
    
    df_filtered.drop(columns=['soil_erosi', 'soil_erosi_no_decimal'], inplace=True)

# Prepare features and target
excluded_cols = ['site_name', 'culture', 'latitude', 'longitude']
feature_cols = [c for c in df_filtered.columns if c not in excluded_cols]
X = df_filtered[feature_cols].copy()
y = df_filtered['culture'].copy()

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
cols_to_one_hot = []
for cat_col in ['veg_type', 'erosion_main']:
    if cat_col in categorical_cols:
        cols_to_one_hot.append(cat_col)

if cols_to_one_hot:
    X = pd.get_dummies(X, columns=cols_to_one_hot, drop_first=False)

for col in categorical_cols:
    if col not in cols_to_one_hot:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

print("\nUpdated Feature columns after encoding:", X.columns.tolist())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Initialize Random Forest
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=cv, scoring='accuracy')

print("\nK-Fold Cross Validation Results (Training set):")
print("Scores per fold: ", cv_scores)
print("Mean CV accuracy: {:.3f}".format(cv_scores.mean()))
print("Std of CV accuracy: {:.3f}".format(cv_scores.std()))

# Train and evaluate
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=["Kayue", "Nuomuhong"])
print("\nConfusion Matrix (Kayue vs. Nuomuhong):")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

# Feature importances
importances = rf_clf.feature_importances_
feat_importance = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
print("\nFeature Importances:")
print(feat_importance)

# Plotting functions
def plot_confusion_matrix(cm, labels, title, ax=None):
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    plt.title(title)
    plt.xlabel("Prediction")
    plt.ylabel("Reference")

def plot_feature_importances(importances, feature_names, title):
    sorted_idx = importances.argsort()[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(
        range(len(sorted_idx)),
        importances[sorted_idx],
        align="center",
        color="gray",
        edgecolor="black"
    )
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel("Mean Gini Decrease")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Confusion matrix plot
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
labels = ["Kayue", "Nuomuhong"]

plt.figure(figsize=(8, 6))
plot_confusion_matrix(cm_normalized, labels, title="Confusion Matrix (Test Set)")
plt.show()

# Feature importances plot
plot_feature_importances(rf_clf.feature_importances_, X_train.columns, "Feature Importances (Mean Gini Decrease)")

# Optional OOB confusion matrix
if rf_clf.oob_score:
    y_oob_pred = rf_clf.oob_decision_function_.argmax(axis=1)
    oob_cm = confusion_matrix(y_train.map({"Kayue": 0, "Nuomuhong": 1}), y_oob_pred)
    oob_cm_normalized = oob_cm.astype('float') / oob_cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(oob_cm_normalized, labels, title="OOB Confusion Matrix")
    plt.show()

# Violin plots for selected variables
variables_to_plot = {
    "elevation": "Elevation",
    "NDVI": "NDVI",
    "precipitation": "Precipitation",
    "soil_type": "Soil Type",
    "cultivated": "Cultivated Land Suitability"
}

num_vars = len(variables_to_plot)
cols = 2
rows = (num_vars + 1) // cols

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(18, rows * 6),
    constrained_layout=True
)

axes = axes.flatten()
palette = sns.color_palette("Blues", n_colors=2)

for i, (var, display_name) in enumerate(variables_to_plot.items()):
    sns.violinplot(
        data=df_filtered,
        x="culture",
        y=var,
        inner="box",
        palette=palette,
        ax=axes[i]
    )
    axes[i].set_title(
        f"Distribution of {display_name} by Culture",
        fontsize=14,
        pad=10
    )
    axes[i].set_xlabel("Culture", fontsize=12)
    axes[i].set_ylabel(display_name, fontsize=12)
    axes[i].tick_params(axis='both', labelsize=10)

# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle(
    "Violin Plots of Various Features by Culture",
    fontsize=18,
    y=1.03
)

plt.show()
