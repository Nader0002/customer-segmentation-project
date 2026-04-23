import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_data(path):
    df = pd.read_csv(path)
    df.columns = ["CustomerID", "Gender", "Age", "AnnualIncome", "SpendingScore"]
    df = df.drop_duplicates().dropna()
    return df

def prepare_features(df):
    X = df[["AnnualIncome", "SpendingScore"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler

def find_best_k(X_scaled):
    best_k = None
    best_score = -1

    for k in range(2, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score

def train_model(X_scaled, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    return model, labels

def main():
    df = load_data(r"C:\Users\NaderV2\Desktop\customer-segmentation-project\data\mall_customers.csv")
    X, X_scaled, scaler = prepare_features(df)
    best_k, best_score = find_best_k(X_scaled)

    model, labels = train_model(X_scaled, best_k)
    df["Cluster"] = labels

    print("Best k:", best_k)
    print("Best silhouette score:", best_score)
    print(df.groupby("Cluster")[["Age", "AnnualIncome", "SpendingScore"]].mean())

    plt.figure(figsize=(8, 6))
    plt.scatter(df["AnnualIncome"], df["SpendingScore"], c=df["Cluster"])
    plt.title("Customer Segments")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.show()

if __name__ == "__main__":
    main()