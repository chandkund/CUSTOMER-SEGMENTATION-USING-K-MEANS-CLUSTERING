# CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING

## Project Overview

This project focuses on customer segmentation using the K-Means Clustering algorithm. By clustering customers based on their purchasing behaviors and demographics, businesses can tailor their marketing strategies and improve customer engagement.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Model Evaluation](#model-evaluation)
- [License](#license)

## Installation

To run this project, you will need Python along with the following libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install the required packages using `pip`:

    pip install pandas numpy matplotlib seaborn scikit-learn

Clone the repository:

    git clone https://github.com/chandkund/customer-segmentation-using-k-means-clustering.git
    cd customer-segmentation-using-k-means-clustering

## Usage

- Prepare the Dataset:
  Place your dataset (e.g., `mall_customers.csv`) in the project directory or adjust the file path in the code.

- Run the Code:
  Execute the Python scripts to perform data preprocessing, clustering, and visualization.

      python script.py

## Code Explanation

- **Import Relevant Libraries**:

    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans
    ```

- **Load and Preprocess Data**:

    ```python
    raw_data = pd.read_csv("path/to/your/dataset.csv")
    df = raw_data.copy()
    ```

- **Visualize Data Before Normalization**:

    ```python
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df)
    plt.show()
    ```

- **Normalize the Data**:

    ```python
    cols = ['Annual Income (k$)', 'Spending Score (1-100)']
    scaled = MinMaxScaler()
    df[cols] = pd.DataFrame(scaled.fit_transform(df[cols]), columns=cols)
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df)
    plt.show()
    ```

- **Determine Optimal Number of Clusters**:

    ```python
    Wcss = []
    for i in range(1, 12):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(df[cols])
        Wcss.append(kmeans.inertia_)

    plt.plot(range(1, 12), Wcss)
    plt.title("The Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Wcss Values")
    plt.show()
    ```

- **Train the K-Means Model**:

    ```python
    optimal_clusters = 5
    kmeans_model = KMeans(n_clusters=optimal_clusters, random_state=0)
    df['cluster'] = kmeans_model.predict(df[cols])
    ```

- **Visualize Clusters**:

    ```python
    plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['cluster'], cmap='rainbow')
    plt.title("Clusters of Customers")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.show()
    ```

## Model Evaluation

Evaluate the clustering results by analyzing the characteristics of each cluster and the distribution of data points within each cluster.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
