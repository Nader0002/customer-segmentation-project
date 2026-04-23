# Customer Segmentation using K-Means

## Project Overview
This project applies unsupervised machine learning (K-Means clustering) to segment customers based on income and spending behavior.

## Dataset
- 200 customers
- Features: Age, Annual Income, Spending Score

## Methodology
- Data cleaning & preprocessing
- Feature selection
- Elbow Method & Silhouette Score for optimal K
- K-Means clustering

## Results
- Identified 5 customer segments:
  - VIP customers (high income, high spending)
  - High-potential customers (high income, low spending)
  - Low-value customers
  - etc.

## Outputs
- Cluster visualization
- Cluster summary (Excel)

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Scikit-learn

## How to Run
```bash
pip install -r requirements.txt
python notebooks/main.py