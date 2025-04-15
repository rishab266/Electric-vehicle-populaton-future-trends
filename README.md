# Electric Vehicle Range Prediction

This project aims to predict the electric range of vehicles using machine learning. 

## Project Description

This project analyzes a dataset of electric vehicles to understand the factors that influence their electric range. It uses data exploration, feature engineering, and machine learning models to build a predictive model. The primary goal is to achieve accurate predictions of the electric range based on vehicle characteristics.

## Dataset

The dataset used in this project is "Electric_Vehicle_Population_Data.csv". It contains information about electric vehicles, including make, model, year, electric range, battery capacity, and other relevant features.

## Methodology

The project follows these steps:

1. **Data Loading and Exploration:** Load the dataset into a pandas DataFrame and explore its characteristics using descriptive statistics, visualizations, and identifying missing values.
2. **Feature Engineering:** Create new features from the existing data to potentially improve model performance. This includes creating an 'Age' feature, categorizing 'Electric Range', and one-hot encoding categorical features.
3. **Data Preparation:** Prepare the data for modeling by selecting relevant features, handling missing values, and scaling numerical features.
4. **Data Splitting:** Split the prepared data into training and testing sets to avoid data leakage and evaluate model performance.
5. **Model Training:** Train a machine learning model (e.g., Linear Regression, Random Forest) on the training data to predict the electric range.
6. **Model Evaluation:** Evaluate the model's performance using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared.
7. **Model Optimization:** Refine the model through feature engineering, hyperparameter tuning, and exploring alternative models to improve prediction accuracy.

## Findings

* **Market Dominance:** Tesla and the Model Y are the most frequent make and model in the dataset, indicating their significant market share in the electric vehicle industry.
* **Vehicle Type:** Battery Electric Vehicles (BEVs) are more prevalent than Plug-in Hybrid Electric Vehicles (PHEVs) in the dataset.
* **Geographic Distribution:** California and other western states have a higher concentration of electric vehicles compared to other regions in the US.
* **Model Performance:** While the initial Linear Regression model showed poor performance (negative R-squared), using alternative models like Random Forest, along with proper data preprocessing, led to improvements in prediction accuracy.
* **Feature Importance:** The 'Model Year', 'Age', 'Base MSRP', and 'Electric Vehicle Type' appear to be significant factors influencing electric range predictions. Further analysis can reveal more detailed feature importance rankings.
* **Data Challenges:** The dataset contains missing values in crucial columns like 'Electric Range', requiring careful imputation strategies. Outliers in 'Base MSRP' and 'Electric Range' were observed, impacting model performance and potentially requiring specialized handling.


## Future Predictions and Applications

This project has the potential to be extended and applied in various ways:

* **Real-time Range Estimation:** Integrate the model into electric vehicle navigation systems or mobile apps to provide real-time range estimates based on current driving conditions and vehicle specifications.
* **Vehicle Design and Optimization:** Utilize the model to guide the design and optimization of future electric vehicles by identifying key features that significantly impact range.
* **Charging Infrastructure Planning:** Leverage the model's predictions to assist in planning and optimizing the placement of charging infrastructure to meet the needs of electric vehicle users.
* **Market Analysis and Forecasting:** Analyze trends in electric vehicle range and market demand to support business decisions related to production, pricing, and marketing strategies.
* **Environmental Impact Assessment:** Incorporate the model into studies assessing the environmental impact of electric vehicles, considering factors like energy consumption and emissions.


## Results

The initial results obtained using a linear regression model were unsatisfactory, indicated by a negative R-squared value. This led to further exploration and optimization strategies. Improvements were made by using alternative models such as Random Forest and careful handling of the data preparation and splitting steps. Further optimization strategies can be explored to improve the prediction accuracy.

## Conclusion

This project demonstrates the process of building a predictive model for electric vehicle range. The optimization process is ongoing, and further refinements could be made. The project highlights the importance of data preparation, feature engineering, and model selection in achieving accurate predictions.


## Usage

1. Clone this repository: `git clone <repository_url>`
2. Open the Colab notebook: `Electric_Vehicle_Range_Prediction.ipynb`
3. Run the notebook cells to execute the analysis and modeling steps.

## Dependencies

- Python 3
- pandas
- scikit-learn
- matplotlib
- seaborn
- plotly

Install the necessary libraries using: `pip install pandas scikit-learn matplotlib seaborn plotly`

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues for discussion.

## License

This project is licensed under the MIT License.
