# SpaceX Launch Success Prediction Project 🚀

## Project Overview

This project focuses on predicting the success of SpaceX launches based on historical launch data. The objective is to build a machine learning model that can determine the likelihood of a successful launch. The project showcases advanced data science techniques, including data wrangling, web scraping, exploratory data analysis (EDA), and multiple machine learning models to provide an accurate solution to the business problem.

## Problem Statement

SpaceX aims to make space exploration more affordable through the reuse of rocket boosters. The goal of this project is to predict the outcome (success/failure) of these launches based on various factors like orbit, payload, and landing outcomes.

By analyzing historical data and predicting launch success rates, SpaceX can enhance decision-making for future missions, optimizing costs and improving mission planning.

## Key Features

- **Data Wrangling:** Cleaning and preprocessing the dataset to handle missing values, inconsistencies, and ensuring the data is in a usable format.
  
- **Web Scraping:** Data collection from public resources using BeautifulSoup and requests to gather up-to-date SpaceX launch data.

- **Exploratory Data Analysis (EDA):** Advanced data visualization using libraries like Matplotlib and Seaborn to understand patterns in launch success based on different features such as orbit, payload mass, and booster version.

- **Machine Learning Models:**
  - Logistic Regression
  - Decision Tree Classifier
  - K-Nearest Neighbors (KNN)

  Models were optimized using GridSearchCV to find the best parameters for predicting the success of SpaceX launches.

- **Model Evaluation:** Metrics such as accuracy, precision, recall, F1-score, and AUC-ROC were used to evaluate the performance of the models.

## Business Insights

The project provides valuable business insights by identifying key factors influencing SpaceX launch success rates. These insights can assist SpaceX in optimizing mission outcomes, minimizing the cost of rocket launches, and increasing the reliability of missions.

## Technologies Used

- **Languages:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, BeautifulSoup, Requests
- **Machine Learning Techniques:**
  - Logistic Regression
  - Decision Tree Classifier
  - K-Nearest Neighbors
  - GridSearchCV for hyperparameter tuning

## Project Workflow

1. **Data Collection:**
   - Web scraping SpaceX launch data from publicly available sources.
   - Cleaning and preprocessing the data to remove inconsistencies.

2. **Exploratory Data Analysis:**
   - Visualizing and analyzing relationships between different features and launch success.
   - Key factors like orbit type, payload mass, and booster type were identified.

3. **Feature Engineering:**
   - Creating new features to improve the model's accuracy.

4. **Model Building:**
   - Using Logistic Regression, Decision Trees, and KNN models.
   - Hyperparameter tuning with GridSearchCV to improve performance.

5. **Evaluation:**
   - Evaluating models using classification metrics and choosing the best-performing model.

## Results

The best-performing model achieved an accuracy of X% on the test set. The analysis highlighted the significant impact of certain features, such as orbit type and payload mass, on the success rate of SpaceX launches.

## Conclusion

This project demonstrates the application of various data science and machine learning techniques to solve a real-world business problem. By predicting SpaceX launch success, the project helps SpaceX optimize its future launches, reduce costs, and improve mission reliability.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/spacex-launch-prediction.git
2. Navigate to the project folder:
   ```bash
   cd spacex-launch-prediction
3. Run the jupyter notebooks and learn
