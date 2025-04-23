# Titanic Survival Prediction - Machine Learning Project

This project uses the **Titanic Dataset by M Yasser.H** to predict the survival of passengers using various machine learning algorithms. The implementation is done in a Jupyter Notebook (`TSP.ipynb`).

## 🚀 Objective

To build a predictive model that determines whether a passenger survived the Titanic disaster based on features like age, sex, passenger class, etc.

---

## 📁 Files

- `TSP.ipynb`: The main notebook containing all preprocessing, model training, and evaluation steps.
- `TitanicDataset.csv`: The dataset used for training and evaluation (not included here for size/privacy; please add your own).

---

## 🧪 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn (optional for visualization)

---

## 🔍 Steps Involved

1. **Data Loading**
   - Load CSV data using Pandas

2. **Data Preprocessing**
   - Handle missing values
   - Encode categorical features
   - Feature selection

3. **Model Training**
   - Train using Logistic Regression / Decision Trees / Random Forests

4. **Model Evaluation**
   - Accuracy score
   - Confusion matrix

5. **Prediction**
   - Run model on test data or user-input values (if available)

---

## 📊 Features Used

- `Pclass` (Passenger Class)
- `Age`
- `SibSp` (Number of Siblings/Spouses aboard)
- `Parch` (Number of Parents/Children aboard)
- `Fare`
- `Sex_male` (Binary encoded)
- `Embarked_Q`, `Embarked_S` (One-hot encoded)

---

## ⚠️ Notes

- Make sure the dataset path is correct when running the notebook.
- If you get a warning about convergence, increase the number of iterations using `max_iter`.
- If you encounter missing column errors, re-run the encoding or reload the dataset.

---

## 🙌 Acknowledgements

- Dataset: [Kaggle - Titanic Dataset by M Yasser.H](https://www.kaggle.com)
- Libraries: Scikit-learn, Pandas, NumPy

---

## 💡 Future Work

- Improve accuracy with hyperparameter tuning
- Add more models (like SVM, XGBoost)
- Use cross-validation
- Deploy model using Flask or Streamlit

