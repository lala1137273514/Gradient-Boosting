# Model Comparison for Test Case Classification

## Objective:
To classify test cases into **Positive (Pos)** or **Negative (Neg)** categories using supervised machine learning algorithms.

---

## Dataset Overview:
- **Source**: A labeled test case dataset with two columns:
  - **Label**: Classification label (Pos or Neg).
  - **Text**: The description of the test case.

### Preprocessing:
- Labels are mapped to binary values: **1 for Pos** and **0 for Neg**.
- Text is converted into numerical features using the **CountVectorizer**.

---

## Algorithms Compared:
1. **Random Forest** - 89% accuracy  
2. **Multinomial Naive Bayes** - 80% accuracy  
3. **Decision Tree** - 92% accuracy  
4. **Logistic Regression** - 87% accuracy  
5. **Support Vector Machine (SVM)** - 87% accuracy  

---

## New Algorithm - Gradient Boosting:
- **Model Used**: `GradientBoostingClassifier`  
- **Accuracy Achieved**: **94.56%**  
- **Improvement**: Gradient Boosting outperformed all the previous algorithms.

---

## Steps Taken:
1. Loaded and cleaned the dataset.  
2. Split the data into **70% training** and **30% testing**.  
3. Converted the text data into numerical form using the **CountVectorizer**.  
4. Trained the **Gradient Boosting Classifier** and evaluated its accuracy.  
5. Compared the results with the other algorithms.  

---

## Conclusion:
The **Gradient Boosting Classifier** is the most effective model for this dataset, achieving an accuracy of **94.56%**, surpassing all five previously used models.


