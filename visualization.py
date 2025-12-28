import seaborn as sns
import matplotlib.pyplot as plt

def plot_clusters(df):
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        hue='Cluster',
        data=df,
        palette='Set1'
    )
    plt.title("Customer Segments")
    plt.show()
