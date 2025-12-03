import matplotlib.pyplot as plt
import seaborn as sns

def plot_churn_distribution(df):
    sns.countplot(x='Churn', data=df)
    plt.title("Churn Distribution")
    plt.show()

def plot_correlation(df):
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
