"""
This is the template file for the
Statistics and Trends assignment.
You are expected to complete all sections and
make this a fully functional, documented file.
You should NOT change any function, file,
or variable names if they are provided here.
Make use of the functions introduced in the lectures,
and ensure your code follows PEP-8 guidelines,
including proper docstrings.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

# Dataset link is given below
# https://www.kaggle.com/datasets/ankushpanday1/heart-attack-in-youth-vs-adult-in-france


def plot_relational_plot(df):
    """
    For relational plot we will create line
    graph means we will show our data on it
    This function shows the link between 'Age' and 'Weight_kg'
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    average_weight = df.groupby('Age')['Weight_kg'].mean()
    plt.plot(average_weight.index, average_weight.values,
             marker='o', color='b', linestyle='-', label='Line Graph')
    plt.title('An Average Weight by Age: Line Graph')
    plt.xlabel('Age in Years')
    plt.ylabel('Average Weight in Kilograms')
    plt.legend()
    plt.savefig('relational_plot.png')
    plt.clf()


def plot_categorical_plot(df):
    """
    For categorical plot we will create a bar graph
    means we will show our data on it.
    This shows we the  distribution of 'Sex' in the dataset
    """
    fig, ax1 = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, x='Sex', ax=ax1)
    plt.title('A Distribution of Gender shown by Bar Graph')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.savefig('categorical_plot.png')
    plt.clf()


def plot_statistical_plot(df):
    """
    For statistical plot we will create violin graph
    means we will show our data on it
    This shows shows the link link between 'BMI' and 'Sex'
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(data=df, x='Sex', y='BMI', ax=ax)
    plt.title('A Violin Plot of BMI by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Body Mass Index (BMI)')
    plt.savefig('statistical_plot.png')
    plt.clf()


def statistical_analysis(df, col: str):
    """
    Calculates the statistical moments for a given column,
    including the mean, standard deviation, skewness, and kurtosis.

    Here we have 2 arguments df means datframe,
    col means column and simlarly we return these all.
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

    Here we have only 1 argument df means dataframe and same the return
    """
    print("The dataset's first five rows are:")
    print(df.head())

    print("\nBefore handling missing values, the number of rows is:", len(df))

    # In Following lines we are handling missing values
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].mean())

    for col in df.select_dtypes(exclude=np.number).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("\nAfter handling missing values, the number of rows is:", len(df))
    print("\nMissing value strategy specifics: "
          "Numerical columns were filled with their mean, "
          "and categorical columns with their mode.")

    print("\nDescriptive Statistics:")
    print(df.describe())

    print("\nCorrelation Matrix:")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()
    print(corr_matrix)

    print("\nFindings:")
    print(" - Strong positive relationship between "
          "Systolic & Diastolic Blood Pressure.")
    print(" - Moderate correlation between Cholesterol Level and Age.")

    return df


def writing(moments, col):
    """
    The statistical moments are shown along with their definitions.

    Arguments:
        moments: It nncludes mean, standard deviation, skewness,
        and kurtosis.
        column: Indicating the column being examined.
    """
    mean, sd, skewnes, kurtosis = moments

    print('\nFor the column ' + str(col) + ':')
    print('Mean = ' + str(round(mean, 2)))
    print('Standard Deviation = ' + str(round(sd, 2)))
    print('Skewness = ' + str(round(skewnes, 2)))
    print('Excess Kurtosis = ' + str(round(kurtosis, 2)))

    # Following are the conditions
    if skewnes > 0:
        print('Data is positively right skewed '
              '(means has a longer right tail).')
    elif skewnes < 0:
        print('Data is negatively left skewed'
              ' (means has a longer left tail).')
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


def main():
    """
    Main purpose of this function is to load, clean, evaluate,
    and display data
    """
    df = pd.read_csv('data.csv')
    df = preprocessing(df)

    # Here we are choosing the BMI as our selection
    # on which basis analysis will done
    col = 'BMI'
    print("\nSelected column for analysis: BMI")
    print("More information on BMI: An important health marker"
          "often used in medical research. It calculates "
          "body fat from weight and height.")

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == '__main__':
    main()
