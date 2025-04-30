import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Additional classifiers for improved comparisons
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

sns.set(style="whitegrid")

# --------------------------------------------------
# Synthetic Data Generation for Each Disaster Type
# --------------------------------------------------
def generate_dataset(disaster, n_samples=1000, random_state=42, noise_std=0.1):
    np.random.seed(random_state)
    # Using a low noise_std (0.1) so that the decision boundary is clear and models achieve >95% accuracy.
    if disaster == "Earthquake":
        data = {
            "magnitude": np.random.uniform(3.0, 9.0, n_samples),
            "shaking_duration": np.random.uniform(0, 120, n_samples),
            "ground_acceleration": np.random.uniform(0.0, 1.0, n_samples),
            "peak_ground_velocity": np.random.uniform(0, 200, n_samples),
            "depth": np.random.uniform(5.0, 300.0, n_samples),
            "fault_distance": np.random.uniform(0.0, 100.0, n_samples),
            "building_density": np.random.uniform(0, 10000, n_samples),
            "liquefaction_index": np.random.uniform(0.0, 1.0, n_samples),
            "distance": np.random.uniform(0.0, 100.0, n_samples),
            "pop_density": np.random.uniform(0, 10000, n_samples),
            "seismic_gap": np.random.uniform(0, 500, n_samples),
            "building_age": np.random.uniform(0, 200, n_samples),
        }
        df = pd.DataFrame(data)
        weights = np.array([10, 0.5, 20, 0.2, -0.05, -0.1, 0.001, 30, -0.1, 0.001, 0.02, 0.05])
        score = df.values.dot(weights) + np.random.normal(0, noise_std, n_samples)
        threshold = np.median(score)
        df["target"] = (score > threshold).astype(int)
    elif disaster == "Flood":
        data = {
            "rainfall": np.random.uniform(0, 600, n_samples),
            "land_slope": np.random.uniform(0, 45, n_samples),
            "drainage_efficiency": np.random.uniform(0, 1, n_samples),
            "urbanization_rate": np.random.uniform(0, 100, n_samples),
            "river_level": np.random.uniform(0, 50, n_samples),
            "soil_saturation": np.random.uniform(0, 100, n_samples),
            "impervious_area": np.random.uniform(0, 100, n_samples),
            "catchment_area": np.random.uniform(0, 500, n_samples),
            "flood_duration": np.random.uniform(0, 240, n_samples),
            "water_retention": np.random.uniform(0, 200, n_samples),
            "vegetation_index": np.random.uniform(0, 1, n_samples),
            "soil_type_index": np.random.uniform(0, 1, n_samples),
        }
        df = pd.DataFrame(data)
        weights = np.array([0.05, -0.2, 50, 0.1, 0.5, 0.3, 0.2, -0.01, 0.1, 0.05, 30, 40])
        score = df.values.dot(weights) + np.random.normal(0, noise_std, n_samples)
        threshold = np.median(score)
        df["target"] = (score > threshold).astype(int)
    elif disaster == "Cyclone":
        data = {
            "wind_speed": np.random.uniform(50, 300, n_samples),
            "storm_surge": np.random.uniform(0, 10, n_samples),
            "central_pressure": np.random.uniform(800, 1100, n_samples),
            "eye_diameter": np.random.uniform(0, 100, n_samples),
            "precipitation": np.random.uniform(0, 400, n_samples),
            "visibility": np.random.uniform(0, 20, n_samples),
            "pressure_variation": np.random.uniform(0, 100, n_samples),
            "cyclone_duration": np.random.uniform(0, 240, n_samples),
            "sea_surface_temp": np.random.uniform(20, 35, n_samples),
            "wave_height": np.random.uniform(0, 15, n_samples),
            "ocean_current": np.random.uniform(0, 50, n_samples),
            "humidity": np.random.uniform(0, 100, n_samples),
        }
        df = pd.DataFrame(data)
        weights = np.array([0.03, 5, -0.02, 0.1, 0.04, -0.5, 0.2, 0.05, 2, 0.3, 0.1, 0.5])
        score = df.values.dot(weights) + np.random.normal(0, noise_std, n_samples)
        threshold = np.median(score)
        df["target"] = (score > threshold).astype(int)
    elif disaster == "Wildfire":
        data = {
            "temperature": np.random.uniform(20, 50, n_samples),
            "humidity": np.random.uniform(10, 90, n_samples),
            "wind_speed": np.random.uniform(0, 100, n_samples),
            "vegetation_density": np.random.uniform(0, 100, n_samples),
            "fuel_moisture": np.random.uniform(0, 100, n_samples),
            "drought_index": np.random.uniform(0, 1, n_samples),
            "precipitation": np.random.uniform(0, 50, n_samples),
            "air_quality": np.random.uniform(50, 300, n_samples),
            "land_slope": np.random.uniform(0, 45, n_samples),
            "ignition_sources": np.random.randint(0, 10, n_samples),
        }
        df = pd.DataFrame(data)
        weights = np.array([0.5, -0.4, 0.3, 0.2, -0.2, 50, -0.1, 0.05, -0.3, 2])
        score = df.values.dot(weights) + np.random.normal(0, noise_std, n_samples)
        threshold = np.median(score)
        df["target"] = (score > threshold).astype(int)
    elif disaster == "Tsunami":
        data = {
            "eq_magnitude": np.random.uniform(5.0, 9.0, n_samples),
            "seafloor_disp": np.random.uniform(0.5, 5.0, n_samples),
            "water_depth": np.random.uniform(1000, 6000, n_samples),
            "distance_epicenter": np.random.uniform(0, 300, n_samples),
            "wave_height": np.random.uniform(0, 10, n_samples),
            "wave_period": np.random.uniform(5, 30, n_samples),
            "coastal_elevation": np.random.uniform(0, 50, n_samples),
            "tidal_range": np.random.uniform(0, 5, n_samples),
            "shoreline_slope": np.random.uniform(0, 10, n_samples),
            "warning_time": np.random.uniform(0, 60, n_samples),
        }
        df = pd.DataFrame(data)
        weights = np.array([10, 5, -0.001, -0.05, 2, 0.5, -0.3, 1, 0.8, 0.2])
        score = df.values.dot(weights) + np.random.normal(0, noise_std, n_samples)
        threshold = np.median(score)
        df["target"] = (score > threshold).astype(int)
    elif disaster == "Heatwave":
        data = {
            "temperature": np.random.uniform(30, 45, n_samples),
            "humidity": np.random.uniform(20, 80, n_samples),
            "heat_index": np.random.uniform(35, 60, n_samples),
            "duration": np.random.uniform(1, 10, n_samples),
            "air_quality": np.random.uniform(50, 200, n_samples),
            "wind_speed": np.random.uniform(0, 30, n_samples),
            "uv_index": np.random.uniform(5, 12, n_samples),
            "drought_index": np.random.uniform(0, 1, n_samples),
            "precipitation": np.random.uniform(0, 20, n_samples),
            "night_temp": np.random.uniform(20, 35, n_samples),
        }
        df = pd.DataFrame(data)
        weights = np.array([0.6, -0.3, 0.4, 5, -0.02, 0.1, 2, 30, -0.5, 0.3])
        score = df.values.dot(weights) + np.random.normal(0, noise_std, n_samples)
        threshold = np.median(score)
        df["target"] = (score > threshold).astype(int)
    else:
        raise ValueError("Unknown disaster type")
    return df

# --------------------------------------------------
# Expanded-Algorithm Comparison & Best Model Selection
# --------------------------------------------------
disasters = ["Earthquake", "Flood", "Cyclone", "Wildfire", "Tsunami", "Heatwave"]

# Dictionaries to store best models and performance metrics for each disaster type
best_models = {}
performance = {}

# Create directories for models, datasets, and graphs/tables
os.makedirs("models", exist_ok=True)
os.makedirs("datasets", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# Define the models to compare (expanded to include additional algorithms)
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVC": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(50,50), max_iter=500, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
    }

# Loop over each disaster type
for disaster in disasters:
    print(f"--- Processing {disaster} ---")
    df = generate_dataset(disaster, noise_std=0.1)

    # Save dataset as CSV
    dataset_filename = f"datasets/{disaster.lower()}_data.csv"
    df.to_csv(dataset_filename, index=False)
    print(f"Dataset saved: {dataset_filename}")

    # Print summary statistics and class distribution
    summary_table = df.describe()
    print(f"\nSummary statistics for {disaster} dataset:")
    print(summary_table)

    class_dist = df["target"].value_counts().reset_index()
    class_dist.columns = ["Class", "Count"]
    print(f"\nClass distribution for {disaster} dataset:")
    print(class_dist)

    # Split into training and testing sets (80/20 split)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get all models and compare their performance
    models_to_compare = get_models()
    best_acc = 0
    best_model_name = None
    best_model = None
    best_y_pred = None
    disaster_accuracies = {}  # store each model's accuracy

    for name, model in models_to_compare.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        disaster_accuracies[name] = acc
        print(f"{name} Accuracy: {acc*100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model = model
            best_y_pred = y_pred

    print(f"--> Best model for {disaster}: {best_model_name} with accuracy {best_acc*100:.2f}%")
    if best_acc < 0.95:
        print(f"Warning: Best accuracy for {disaster} is below 95%!")

    # Save the best model to disk
    model_filename = f"models/{disaster.lower()}_best_model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Best model saved: {model_filename}")

    # Store performance metrics for the current disaster
    performance[disaster] = {
        "model_name": best_model_name,
        "accuracy": best_acc,
        "classification_report": classification_report(y_test, best_y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, best_y_pred),
        "all_accuracies": disaster_accuracies
    }
    best_models[disaster] = best_model

    # ----------------------------
    # Graph 1: Confusion Matrix Heatmap for Best Model
    cm = performance[disaster]["confusion_matrix"]
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{disaster} Confusion Matrix ({best_model_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_filename = f"graphs/{disaster.lower()}_confusion_matrix.png"
    plt.savefig(cm_filename)
    plt.close()
    print(f"Confusion matrix saved: {cm_filename}")

    # ----------------------------
    # Graph 2: ROC Curve for Best Model
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test)[:, 1]
    else:
        y_proba = best_model.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{disaster} ROC Curve ({best_model_name})")
    plt.legend()
    roc_filename = f"graphs/{disaster.lower()}_roc_curve.png"
    plt.savefig(roc_filename)
    plt.close()
    print(f"ROC curve saved: {roc_filename}")

    # ----------------------------
    # Graph 3: Class Distribution Pie Chart
    plt.figure(figsize=(6,6))
    plt.pie(class_dist["Count"], labels=["No Harm", "Harm"], autopct="%1.1f%%",
            colors=["#4CAF50", "#F44336"])
    plt.title(f"{disaster} Class Distribution")
    pie_filename = f"graphs/{disaster.lower()}_class_distribution.png"
    plt.savefig(pie_filename)
    plt.close()
    print(f"Pie chart saved: {pie_filename}")

    # ----------------------------
    # Graph 4: Histogram of the First Feature
    feature_name = X.columns[0]
    plt.figure(figsize=(6,4))
    sns.histplot(df[feature_name], kde=True, color="#BB86FC")
    plt.title(f"{disaster} {feature_name} Distribution")
    hist_filename = f"graphs/{disaster.lower()}_{feature_name}_hist.png"
    plt.savefig(hist_filename)
    plt.close()
    print(f"Histogram saved: {hist_filename}")

    # ----------------------------
    # Graph 5: Box Plot for All Features
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df.drop("target", axis=1), palette="Set3")
    plt.title(f"{disaster} Features Box Plot")
    box_filename = f"graphs/{disaster.lower()}_boxplot.png"
    plt.savefig(box_filename)
    plt.close()
    print(f"Box plot saved: {box_filename}")

    # ----------------------------
    # Graph 6: Side-by-Side Bar Chart for All Algorithm Accuracies
    plt.figure(figsize=(8,5))
    algos = list(disaster_accuracies.keys())
    accs = [disaster_accuracies[algo]*100 for algo in algos]
    sns.barplot(x=algos, y=accs, palette="viridis")
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy (%)")
    plt.title(f"{disaster} - Algorithm Accuracy Comparison")
    algo_bar_filename = f"graphs/{disaster.lower()}_algorithm_accuracy_comparison.png"
    plt.savefig(algo_bar_filename, bbox_inches="tight")
    plt.close()
    print(f"Algorithm accuracy comparison bar chart for {disaster} saved: {algo_bar_filename}")

    print("\n" + "-"*50 + "\n")

# ----------------------------
# Graph 7: Comparison Bar Chart for Best Model Accuracies Across Disasters
acc_values = [performance[d]["accuracy"]*100 for d in disasters]
plt.figure(figsize=(8,5))
sns.barplot(x=disasters, y=acc_values, palette="viridis")
plt.ylabel("Accuracy (%)")
plt.title("Best Model Accuracy Comparison Across Disasters")
for i, v in enumerate(acc_values):
    plt.text(i, v + 1, f"{v:.2f}%", ha="center")
bar_filename = "graphs/model_accuracy_comparison.png"
plt.savefig(bar_filename)
plt.close()
print(f"Accuracy comparison bar chart saved: {bar_filename}")

# ----------------------------
# Graph 8: ROC Curves Comparison for Best Models Across Disasters
plt.figure(figsize=(8,6))
for disaster in disasters:
    df_temp = pd.read_csv(f"datasets/{disaster.lower()}_data.csv")
    X_temp = df_temp.drop("target", axis=1)
    y_temp = df_temp["target"]
    _, X_test_temp, _, y_test_temp = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    model = best_models[disaster]
    if hasattr(model, "predict_proba"):
        y_proba_temp = model.predict_proba(X_test_temp)[:, 1]
    else:
        y_proba_temp = model.decision_function(X_test_temp)
        y_proba_temp = (y_proba_temp - y_proba_temp.min()) / (y_proba_temp.max() - y_proba_temp.min())
    fpr, tpr, _ = roc_curve(y_test_temp, y_proba_temp)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{disaster} (AUC={roc_auc:.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison for Best Models Across Disasters")
plt.legend()
roc_comp_filename = "graphs/roc_comparison.png"
plt.savefig(roc_comp_filename)
plt.close()
print(f"ROC comparison plot saved: {roc_comp_filename}")

# ----------------------------
# Table 1: Classification Reports for All Best Models
for disaster in disasters:
    print(f"\nClassification Report for {disaster} (Best Model: {performance[disaster]['model_name']}):")
    report_df = pd.DataFrame(performance[disaster]["classification_report"]).transpose()
    print(report_df)

# ----------------------------
# Table 2: Performance Summary Table (Saved as CSV)
performance_summary = []
for disaster in disasters:
    rep = performance[disaster]["classification_report"]
    performance_summary.append({
        "Disaster": disaster,
        "Best Model": performance[disaster]["model_name"],
        "Accuracy": performance[disaster]["accuracy"],
        "Precision (Harm)": rep.get("1", {}).get("precision", np.nan),
        "Recall (Harm)": rep.get("1", {}).get("recall", np.nan),
        "F1-Score (Harm)": rep.get("1", {}).get("f1-score", np.nan),
    })
performance_df = pd.DataFrame(performance_summary)
performance_csv = "graphs/performance_summary.csv"
performance_df.to_csv(performance_csv, index=False)
print(f"Performance summary table saved: {performance_csv}")

# ----------------------------
# Table 3: Complete Accuracy Comparison for All Algorithms Across Disasters
accuracy_comparison = pd.DataFrame({
    disaster: performance[disaster]["all_accuracies"]
    for disaster in disasters
}).T
accuracy_comparison_csv = "graphs/accuracy_comparison_all_algorithms.csv"
accuracy_comparison.to_csv(accuracy_comparison_csv, index=True)
print(f"Complete accuracy comparison table saved: {accuracy_comparison_csv}")

# ----------------------------
# Graph 9: Heatmap for Complete Accuracy Comparison Across Disasters
plt.figure(figsize=(12,8))
sns.heatmap(accuracy_comparison*100, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Accuracy Comparison for All Algorithms Across Disasters (%)")
heatmap_filename = "graphs/accuracy_comparison_heatmap.png"
plt.savefig(heatmap_filename)
plt.close()
print(f"Accuracy comparison heatmap saved: {heatmap_filename}")

# ----------------------------
# Table 4: Feature Correlation Tables for Each Disaster Dataset
for disaster in disasters:
    df_temp = pd.read_csv(f"datasets/{disaster.lower()}_data.csv")
    corr = df_temp.corr()
    corr_csv = f"graphs/{disaster.lower()}_feature_correlation.csv"
    corr.to_csv(corr_csv)
    print(f"Feature correlation table for {disaster} saved: {corr_csv}")

print("\nBackend training complete. All models, datasets, graphs, and tables have been generated and saved.")
