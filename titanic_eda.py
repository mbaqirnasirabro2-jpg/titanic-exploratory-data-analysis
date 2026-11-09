"""
Title: Titanic Dataset Exploratory Data Analysis (EDA) & Visualization Pipeline
Author: Muhammad Baqir (GitHub: @mbaqirnasirabro2-jpg)
Description: Comprehensive EDA script for the Titanic dataset featuring data cleaning,
             missing value imputation, statistical summaries, and professional visualizations (Seaborn & Matplotlib).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set professional visualization theme
sns.set_theme(style="whitegrid")

class TitanicEDAPipeline:
    def __init__(self):
        """Initializes the pipeline and loads the Titanic dataset from Seaborn."""
        print("=== INITIALIZING TITANIC EDA PIPELINE ===")
        self.df = sns.load_dataset('titanic')
        print(f"[SUCCESS] Dataset loaded successfully! Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns\n")

    def inspect_and_clean_data(self):
        """Inspects missing values and cleans the dataset."""
        print("--- 1. Data Inspection & Cleaning ---")
        print("Missing values before cleaning:")
        print(self.df.isnull().sum()[self.df.isnull().sum() > 0])
        print("\n")

        # 1. Fill missing 'age' with median
        median_age = self.df['age'].median()
        self.df['age'].fillna(median_age, inplace=True)
        print(f"Imputed missing 'age' values with median: {median_age}")

        # 2. Drop rows with missing 'embarked' or 'embark_town'
        self.df.dropna(subset=['embarked', 'embark_town'], inplace=True)
        print("Cleaned missing embarkation records.")
        print("Data cleaning completed successfully!\n" + "-" * 40 + "\n")

    def print_statistical_summary(self):
        """Prints core statistical summaries of the dataset."""
        print("--- 2. Statistical Summary ---")
        print(self.df.describe(include='all').to_string())
        print("\n" + "-" * 40 + "\n")

    def generate_visualizations(self):
        """Generates a 2x2 grid of professional EDA graphs and saves them as an image."""
        print("--- 3. Generating Exploratory Graphs ---")
        
        # Create a figure with 4 subplots (2x2 grid)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Titanic Dataset Exploratory Data Analysis (EDA)', fontsize=16, fontweight='bold', y=1.02)

        # 1. Plot: Passenger Survival Count
        sns.countplot(data=self.df, x='survived', ax=axes[0, 0], palette='Set2')
        axes[0, 0].set_title('Passenger Survival Count (0 = No, 1 = Yes)', fontweight='semibold')
        axes[0, 0].set_xlabel('Survived')
        axes[0, 0].set_ylabel('Passenger Count')

        # 2. Plot: Survival by Passenger Class
        sns.countplot(data=self.df, x='pclass', hue='survived', ax=axes[0, 1], palette='Set1')
        axes[0, 1].set_title('Survival Count by Passenger Class', fontweight='semibold')
        axes[0, 1].set_xlabel('Passenger Class (1 = 1st, 3 = 3rd)')
        axes[0, 1].set_ylabel('Count')

        # 3. Plot: Age Distribution Histogram with KDE
        sns.histplot(data=self.df, x='age', kde=True, ax=axes[1, 0], color='teal', bins=30)
        axes[1, 0].set_title('Passenger Age Distribution', fontweight='semibold')
        axes[1, 0].set_xlabel('Age (Years)')
        axes[1, 0].set_ylabel('Frequency')

        # 4. Plot: Survival by Gender
        sns.countplot(data=self.df, x='sex', hue='survived', ax=axes[1, 1], palette='pastel')
        axes[1, 1].set_title('Survival Count by Gender', fontweight='semibold')
        axes[1, 1].set_xlabel('Gender')
        axes[1, 1].set_ylabel('Count')

        # Adjust layout for neat presentation
        plt.tight_layout()

        # Save plot image to file
        output_image = 'titanic_eda_plots.png'
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Visualizations saved successfully as '{output_image}'!")

        # Display the plots
        plt.show()


# --- Execution Example ---
if __name__ == "__main__":
    # Initialize pipeline
    titanic_pipeline = TitanicEDAPipeline()
    
    # Run Inspection & Cleaning
    titanic_pipeline.inspect_and_clean_data()
    
    # Print Statistics
    titanic_pipeline.print_statistical_summary()
    
    # Generate and Save Graphs
    titanic_pipeline.generate_visualizations()
