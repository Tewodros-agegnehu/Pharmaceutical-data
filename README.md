# **Rossmann Store Sales Prediction**
---

## **Overview**
This project addresses a **machine learning challenge** to predict daily sales for Rossmann Pharmaceuticals stores six weeks ahead of time. Using a combination of **data exploration**, **machine learning models**, and **deep learning techniques**, this project delivers actionable insights and a scalable prediction system for the company's finance team.

---

## **Business Need**
Rossmann Pharmaceuticals' finance team currently relies on store managers’ experience and judgment to forecast sales. This approach is subjective and limits scalability. A data-driven solution is necessary to:
- Improve accuracy in sales forecasting.
- Account for factors like promotions, holidays, seasonality, and local competition.
- Serve predictions as an end-to-end product for the finance team's use.

---

## **Dataset and Features**
The dataset can be accessed from [Kaggle: Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales).

### **Key Features**
- **Id**: Represents a unique `(Store, Date)` pair in the test set.
- **Store**: Unique ID for each store.
- **Sales**: Turnover for any given day (target variable).
- **Customers**: Number of customers on a given day.
- **Open**: Whether the store was open (0 = closed, 1 = open).
- **StateHoliday**: Indicator for state holidays (`a`, `b`, `c`, `0`).
- **SchoolHoliday**: Whether a date was affected by school closures.
- **StoreType**: Differentiates between 4 store models (`a`, `b`, `c`, `d`).
- **Assortment**: Assortment level (`a`, `b`, `c`).
- **CompetitionDistance**: Distance to the nearest competitor store.
- **Promo**: Whether a store is running a promotion.
- **Promo2**: Whether a store participates in a continuous promotion.

---

## **Learning Outcomes**
This project demonstrates proficiency in:
- **Feature Engineering**: Extracting and transforming features for machine learning.
- **Model Building**: Using tree-based and deep learning models for regression tasks.
- **MLOps**: Deploying models with CI/CD pipelines and logging mechanisms.
- **API Development**: Building REST APIs to serve real-time predictions.

---

## **Project Objectives**
1. **Explore Customer Behavior**: Analyze how promotions, holidays, and other factors influence sales.
2. **Predict Store Sales**: Use machine learning and deep learning approaches to forecast sales.
3. **Build and Serve Models**: Develop an API for real-time predictions and deploy it.

---

## **Project Workflow**
1. **Exploratory Data Analysis (EDA)**:
   - Investigate trends, seasonal behaviors, and correlations.
   - Visualize the effects of promotions, holidays, and competition.
2. **Feature Engineering**:
   - Handle missing values and outliers.
   - Generate new features like holiday proximity, weekends, and month positions.
3. **Machine Learning**:
   - Train regression models using Scikit-learn pipelines.
   - Perform hyperparameter tuning for model optimization.
4. **Deep Learning**:
   - Implement an LSTM model to forecast sales using time series data.
5. **Model Serving**:
   - Build a REST API using Flask/FastAPI.
   - Deploy the model for real-time use.

---

## **Installation**
### **Requirements**
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `tensorflow`, `flask`, `joblib`, etc.

### **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/Tewodros-agegnehu/rossmann-sales-prediction.git
   cd rossmann-sales-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the project:
   - For data exploration: Open and run `notebooks/eda.ipynb`.
   - For training: Execute `train_model.py`.
   - For API: Start the server using `app.py`.

---

## **Usage**
1. Train the model:
   ```bash
   python train_model.py
   ```
2. Start the API server:
   ```bash
   python app.py
   ```
3. Make predictions via API:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"Store":1, "DayOfWeek":5, "Promo":1, ...}' http://127.0.0.1:5000/predict
   ```

---

## **Results and Insights**
- **Correlation Analysis**:
  - Positive correlation between promotions and sales volume.
  - Seasonal spikes observed during Christmas and Easter holidays.
- **Model Performance**:
  - Random Forest Regressor: R² = 0.85, RMSE = 1200.
  - LSTM Model: RMSE = 950.
- **Feature Importance**:
  - Top Features: `Promo`, `DayOfWeek`, `CompetitionDistance`.

---

## **Technologies Used**
- **Data Processing**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `RandomForestRegressor`
- **Deep Learning**: `TensorFlow`, `Keras`
- **API Development**: `Flask`
- **Version Control**: `Git`, `DVC`
- **Deployment**: `Heroku`, `AWS`

---

## **Future Improvements**
1. Implement advanced deep learning architectures like GRU.
2. Explore the impact of weather and economic data on sales.
3. Integrate real-time data streams for continuous learning.
4. Enhance deployment using Docker and Kubernetes.

---

## **Acknowledgements**
- [10 Academy](https://10academy.org) for providing the project framework and guidance.
- [Kaggle](https://www.kaggle.com/c/rossmann-store-sales) for the dataset.
- Rossmann Pharmaceuticals for inspiring this sales prediction challenge.

---
