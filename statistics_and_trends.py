"""
This is the template file for the Statistics and Trends assignment.
You are expected to complete all sections and make this a fully functional, documented file.
You should NOT change any function, file, or variable names if they are provided here.
Make use of the functions introduced in the lectures, 
and ensure your code follows PEP-8 guidelines, including proper docstrings.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

# Data source: [Provide the actual link to your dataset, e.g., Kaggle, World Bank]
# Example: Data source: https://www.kaggle.com/datasets/ankushpanday1/heart-attack-in-youth-vs-adult-in-france

# I am going to create 3 graphs for each relational, statistical and catagorical.


def plot_relational_plot(df):
    """
    For relational we will create line, step and scatter plot means we will show our data in these types for relational plot
    """
    # Line Graph
    fig, ax = plt.subplots(figsize=(10, 6))
    average_weight = df.groupby('Age')['Weight_kg'].mean()
    plt.plot(average_weight.index, average_weight.values,
             marker='o', color='b', linestyle='-', label='Line Graph')
    plt.title('An Average weight by age: Line Graph')
    plt.xlabel('Age in years')
    plt.ylabel('Average Wieght in kilograms')
    plt.legend()
    plt.savefig('relational_graph_line.png')
    plt.clf()

    # Step Graph
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.step(average_weight.index, average_weight.values,
             where='mid', label='Step Plot', color='g')
    plt.title('An Average weight by age: Step Graph')
    plt.xlabel('Age in years ')
    plt.ylabel('Average Weight in kilograms')
    plt.legend()
    plt.savefig('relational_graph_step.png')
    plt.clf()

    # Scatter Graph
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='Age', y='Weight_kg',
                    ax=ax, color='r', label='Scatter Graph')
    plt.title('An average weight by age: Scatter Graph')
    plt.xlabel('Age in years)')
    plt.ylabel('Weight in Kilograms')
    plt.legend()
    plt.savefig('relational_graph_scatter.png')
    plt.clf()


def plot_statistical_plot(df):
    """
    For statistical we will show show data in pair,violin and heatmap graph
    """
    # Pair Graph
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    fig = sns.pairplot(numeric_df)
    plt.suptitle('A Pair graph of Numeric Features', y=1.02)
    plt.savefig('statistical_graph_pair.png')
    plt.clf()

    # Heatmap Graph
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    plt.title('Heatmap')
    plt.savefig('statistical_graph_heatmap.png')
    plt.clf()

    # Violin Graph
    fig, ax3 = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x='Sex', y='BMI', ax=ax3)
    plt.title(' A BMI Distribution by Sex: Violin graph ')
    plt.xlabel('Gender')
    plt.ylabel('BMI')
    plt.savefig('statistical_graph_violin.png')


def plot_catagorical_plot(df):
    """
    For catagorical we will show data with Histogram, pie and bar graph.
    """
    # Bar Graph
    fig, ax1 = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, x='Sex', ax=ax1)
    plt.title(' A Distribution of Gender shown by Bar Graph')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.savefig('catagorical_graph_bar.png')
    plt.clf()

    # Pie Chart
    sex_counts = df['Sex'].value_counts()
    fig, ax2 = plt.subplots(figsize=(6, 6))
    ax2.pie(sex_counts, labels=sex_counts.index,
            autopct='%1.1f%%', startangle=90)
    plt.title('A Distribution of Sex shown by Pie Chart)')
    plt.savefig('catagorical_graph_pie.png')
    plt.clf()

    # Histogram
    df['Sex_encoded'] = df['Sex'].map(
        {'Male': 0, 'Female': 1})
    fig, ax3 = plt.subplots(figsize=(8, 6))
    sns.histplot(df['Sex_encoded'], bins=2, kde=False, ax=ax3)
    plt.title('Histogram for Gender Distribution')
    plt.xlabel('Sex (0 = Male, 1 = Female)')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Male', 'Female'])
    plt.savefig('catagorical_graph_histogram.png')
    plt.clf()


def writing(moments, col):
    """
    Here we will show the statistical moments with their explanation

    Here we take 2 arguments, in moments we manage mean, 
    sd, skewness and kurtosis while in col there is column analysis
    """
    mean, sd, skewnes, kurtosis = moments

    print(f'\nFor the column "{col}":')
    print(f'The Mean = {mean:.2f}')
    print(f'The Standard Deviation = {sd:.2f}')
    print(f'The Skewness = {skewnes:.2f}')
    print(f'The Kurtosis = {kurtosis:.2f}')

    # Interpretation
    if skewnes > 0:
        print('Data is positively right skewed (means has a longer right)')
    elif skewnes < 0:
        print('Data is negatively left skewed (means has a longer left)')
    else:
        print('It is approximately symmetric.')

    if kurtosis > 0:
        print('The data that is leptokurtic (heavytailed) have'
              'more extreme values than a normal distribution.')
    elif kurtosis < 0:
        print('The data is platykurtic, or lighttailed, so there'
              'are less extreme values than in a usual distribution.')
    else:
        print('The data tails are like those of a standard'
              'distribution since it is mesokurtic.')


def statistical_analysis(df, col: str):
    """
    Calculates the statistical moments for a given column,
    including the mean, standard deviation, skewness, and kurtosis.

    Here we have 2 arguments df means datframe, col means column
    """
    mean = df[col].mean()
    sd = df[col].std()
    skew = ss.skew(df[col].dropna())
    kurtosis = ss.kurtosis(df[col].dropna())
    return mean, sd, skew, kurtosis


def preprocessing(df):
    """
    Displays statistics, handles missing values, and illustrates 
    correlations as part of the preprocessing step of the dataset.

    Here we have only 1 argument df means dataframe
    """
    print("The dataset's first five rows are:")
    print(df.head())

    print("\nBefore handling missing values, the number of rows is:", len(df))

    # Here we are handling missing values in our data
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].mean())

    for col in df.select_dtypes(exclude=np.number).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("The count of rows is after missing value management:", len(df))
    print("\nMissing value strategy specifics: "
          "Numerical columns were filled with their mean,"
          "and catagorical columns with their mode. ")

    print("\nDescriptive Statistics show:")
    print(df.describe())

    print("\nCorrelation Matrix shiw:")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()
    print(corr_matrix)
    print("\nA strong positive relationship between"
          "Blood Pressure measures (Systolic &amp; Diastolic) seen."
          "Cholesterol Level moderately also relates with Age")

    return df


def main():
    """
    Main purpose of this function is to load, clean, evaluate,
    and display data
    """
    df = pd.read_csv('data.csv')  # Replace with your actual file name
    df = preprocessing(df)

    # Choose a meaningful column for analysis and explain your choice.
    col = 'BMI'  # Example: Body Mass Index
    print("\nSelected column for analysis: BMI")
    print("Further Details about: BMI, a central"
          "health measurement, is often employed in medical research."
          "It calculates body fat estimate from weight and h")

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_catagorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == '__main__':
    main()
