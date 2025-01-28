# Cognifyz Restaurants Study

**Where Data Meets Intelligence**

This repository contains solutions to tasks assigned during the Cognifyz Technologies ML Internship. The focus is on leveraging data science and machine learning techniques to solve problems related to restaurant analysis, including predicting ratings, creating recommendation systems, and performing geographical insights.

---

## Repository Link

Access the project on GitHub: [Cognifyz_Restuarants_Study](https://github.com/mohamed682004/Cognifyz_Restuarants_Study)

---

## Tasks Overview

### **Task 1: Predict Restaurant Ratings**

**Objective**: Develop a machine learning model to predict the aggregate ratings of restaurants based on their features.

#### Steps:
1. **Data Preprocessing**:
   - Handle missing values in the dataset.
   - Encode categorical variables for machine learning.
   - Split the dataset into training and testing sets using `train_test_split`.

2. **Model Development**:
   - Train regression models such as `LinearRegression` or `DecisionTreeRegressor`.

3. **Evaluation**:
   - Use metrics like `mean_squared_error` and `r2_score` to evaluate model performance.

4. **Feature Analysis**:
   - Analyze the importance of features influencing ratings.

---

### **Task 2: Restaurant Recommendation System**

**Objective**: Create a personalized recommendation system for restaurants based on user preferences.

#### Steps:
1. **Data Preparation**:
   - Preprocess the dataset, addressing missing values and encoding categorical variables.

2. **Recommendation Approach**:
   - Use a **content-based filtering** technique based on user-defined criteria like:
     - Cuisine preference
     - Price range

3. **Testing**:
   - Provide sample user preferences and evaluate the quality of recommendations.

---

### **Task 4: Location-Based Analysis**

**Objective**: Perform a geographical analysis of restaurants using their latitude and longitude data.

#### Steps:
1. **Exploration**:
   - Visualize restaurant locations on an interactive map using the `Folium` library.
   - Employ `MarkerCluster` for clustering locations.

2. **Statistical Analysis**:
   - Group restaurants by cities/localities and analyze metrics such as:
     - Average ratings
     - Cuisine distribution
     - Price range

3. **Insights**:
   - Highlight patterns and trends across geographical locations.

---

## Files in the Repository

- **`city_ratings_map.html`**: Interactive map visualizing restaurant ratings by location.
- **`Dataset.csv`**: Dataset used for training models and analysis.
- **`Restaurants_Study.ipynb`**: Jupyter Notebook containing detailed solutions to Tasks 1, 2, and 4.
- **`ML_cover.png`**: Cover image for the documentation.
- **`Internship_Certificate.pdf`**: Certificate for the Cognifyz Internship.

---

## Key Features and Libraries Used

### **Core Libraries**
- `pandas`, `numpy` for data manipulation and preprocessing.

### **Visualization**
- `matplotlib.pyplot`, `seaborn` for static visualizations.
- `WordCloud` for generating word clouds.
- `folium` and `folium.plugins.MarkerCluster` for interactive maps.

### **Data Preprocessing**
- `StandardScaler`, `MinMaxScaler` for feature scaling.
- `LabelBinarizer` for encoding labels.
- `train_test_split` for splitting datasets.

### **Machine Learning**
- Models: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`.
- Deep Learning: Implemented using `tensorflow.keras` with `Sequential` and `Dense` layers.

### **Evaluation Metrics**
- Classification: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`.
- Visualization: `confusion_matrix`, `roc_curve`, `classification_report`.

### **Miscellaneous**
- `parallel_coordinates` for multivariate visualizations.
- Regular expressions for text cleaning and preprocessing.

---

## Insights and Highlights

- **Rating Prediction**:
  - Developed regression models with high predictive accuracy.
  - Identified critical factors affecting restaurant ratings.

- **Recommendation System**:
  - Built a content-based filtering system tailored to user preferences.

- **Geographical Analysis**:
  - Visualized restaurant locations and trends across various cities using Folium.
  - Discovered key patterns in ratings and cuisine popularity.

---

## Getting Started

### Clone the Repository
```bash
git clone https://github.com/mohamed682004/Cognifyz_Restuarants_Study.git
cd Cognifyz_Restuarants_Study
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Notebook
```bash
jupyter notebook Restaurants_Study.ipynb
```

---

## Contribute

Contributions are welcome! If you find any bugs or have suggestions, feel free to open an issue or submit a pull request.

---

## Author

- **Mohamed Ahmed**  
  - Internship at Cognifyz Technologies  
  - [GitHub Profile](https://github.com/mohamed682004)  
