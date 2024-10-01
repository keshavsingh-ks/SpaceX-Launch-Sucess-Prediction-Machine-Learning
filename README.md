SpaceX Launch Success Prediction Using Machine Learning
Project Overview
This project aims to predict the success of SpaceX Falcon 9 launches using historical data. By leveraging various machine learning models, including Logistic Regression, Decision Trees, and k-Nearest Neighbors (KNN), the project identifies critical factors influencing launch outcomes. Data was scraped from Wikipedia, cleaned, and processed for model building. The analysis provides insights that can assist SpaceX in optimizing future launches and increasing success rates.

Business Problem
SpaceX, a leading aerospace manufacturer, strives to improve the success rate of its rocket launches. Each failed launch incurs significant financial losses. Predicting whether a launch will be successful based on historical data is crucial for SpaceX’s operational efficiency. This project addresses this challenge by analyzing previous Falcon 9 launches and building machine learning models to forecast future launch outcomes.

Objectives
The primary objectives of this project are:

Perform web scraping to gather historical data on SpaceX Falcon 9 launches.
Clean and preprocess the data for machine learning model training.
Conduct exploratory data analysis (EDA) to extract meaningful insights.
Build and optimize machine learning models to predict launch success.
Evaluate the model performances and provide actionable insights for SpaceX.
Key Techniques Used
Web Scraping: Data scraped from Wikipedia pages using BeautifulSoup.
Data Wrangling: Cleaned and prepared the dataset, handled missing values, and formatted date/time fields.
Exploratory Data Analysis (EDA): Visualizations used to uncover patterns and relationships in the dataset.
Machine Learning: Built models including Logistic Regression, Decision Trees, and KNN; optimized using GridSearchCV.
Model Evaluation: Evaluated models based on accuracy, precision, recall, F1-score, and confusion matrices.
Project Workflow
1. Data Collection:
Web scraped historical SpaceX launch data from Wikipedia.
Extracted information such as date, launch site, payload, orbit, and outcome.
2. Data Wrangling:
Cleaned the dataset by handling missing values and transforming data types.
Created new features like Launch Site, Booster Version, and Orbit to enrich the dataset.
3. Exploratory Data Analysis:
Analyzed correlations between features and their impact on launch outcomes.
Generated visualizations to identify key trends, such as the effect of payload mass and orbit type on launch success.
4. Machine Learning Models:
Built multiple models to predict the success of launches.
Used GridSearchCV for hyperparameter tuning and cross-validation.
5. Model Evaluation:
Compared models using accuracy, precision, recall, and F1-score.
Selected the best-performing model and assessed its real-world application.
Tools and Libraries
Web Scraping: BeautifulSoup, Requests
Data Processing: Pandas, NumPy
Data Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-learn
Hyperparameter Tuning: GridSearchCV


Project Structure

SpaceX-Launch-Success-Prediction/
│
├── data/                   # Contains the cleaned dataset
├── notebooks/              # Jupyter notebooks for EDA and model building
├── src/                    # Python scripts for data scraping, processing, and model training
├── plots/                  # Visualizations generated during the analysis
├── README.md               # Project overview and instructions
└── requirements.txt        # Dependencies required to run the project


Running the Project

1. Clone the repository:
git clone https://github.com/your_username/SpaceX-Launch-Success-Prediction.git

2. Run the web scraping script to collect the latest SpaceX data (if needed):
python src/web_scraping.py

3. Run the Jupyter notebooks to perform data analysis, model training, and evaluation.








