import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings

# Set plot style for better visualization
plt.style.use('seaborn')
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load and explore the Iris dataset."""
    try:
        # Load Iris dataset
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        
        print("First few rows of the dataset:")
        print(df.head())
        print("\nDataset Info:")
        print(df.info())
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        # Clean dataset (handle missing values if any)
        if df.isnull().any().any():
            df = df.fillna(df.mean(numeric_only=True))
            print("\nMissing values filled with mean.")
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def basic_analysis(df):
    """Perform basic statistical analysis."""
    try:
        print("\nBasic Statistics:")
        print(df.describe())
        
        # Group by species and calculate mean for numerical columns
        print("\nMean measurements by species:")
        print(df.groupby('species').mean())
        
        # Additional observation
        print("\nObservation: The mean measurements vary significantly across species.")
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")

def create_visualizations(df):
    """Create four different visualizations."""
    try:
        # 1. Line chart (cumulative sum of sepal length over index)
        plt.figure(figsize=(10, 6))
        for species in df['species'].unique():
            species_data = df[df['species'] == species]['sepal length (cm)']
            plt.plot(species_data.cumsum(), label=species)
        plt.title('Cumulative Sepal Length by Species')
        plt.xlabel('Sample Index')
        plt.ylabel('Cumulative Sepal Length (cm)')
        plt.legend()
        plt.grid(True)
        plt.savefig('cumulative_sepal_length.png')
        plt.close()
        
        # 2. Bar chart (average petal length by species)
        plt.figure(figsize=(10, 6))
        species_means = df.groupby('species')['petal length (cm)'].mean()
        sns.barplot(x=species_means.index, y=species_means.values)
        plt.title('Average Petal Length by Species')
        plt.xlabel('Species')
        plt.ylabel('Average Petal Length (cm)')
        plt.savefig('avg_petal_length.png')
        plt.close()
        
        # 3. Histogram (sepal width distribution)
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='sepal width (cm)', bins=20, kde=True)
        plt.title('Distribution of Sepal Width')
        plt.xlabel('Sepal Width (cm)')
        plt.ylabel('Count')
        plt.savefig('sepal_width_histogram.png')
        plt.close()
        
        # 4. Scatter plot (sepal length vs petal length)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', 
                       hue='species', size='species', sizes=(50, 200))
        plt.title('Sepal Length vs Petal Length by Species')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.legend()
        plt.savefig('sepal_vs_petal_scatter.png')
        plt.close()
        
        print("\nVisualizations created and saved as PNG files.")
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")

def main():
    """Main function to execute the analysis."""
    df = load_and_explore_data()
    if df is not None:
        basic_analysis(df)
        create_visualizations(df)
    else:
        print("Analysis aborted due to data loading error.")

if __name__ == "__main__":
    main()