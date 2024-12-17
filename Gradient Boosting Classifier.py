from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Preprocessing
X = data['Text']
y = data['Label']

# Convert labels to binary values
y = y.map({'Pos': 1, 'Neg': 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Feature extraction
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train XGBoost Classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_vec, y_train)

# Predictions
y_pred_xgb = xgb.predict(X_test_vec)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

accuracy_xgb

from sklearn.ensemble import GradientBoostingClassifier

# Use Gradient Boosting instead of XGBoost due to performance
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train_vec, y_train)

# Predictions
y_pred_gbc = gbc.predict(X_test_vec)
accuracy_gbc = accuracy_score(y_test, y_pred_gbc)

accuracy_gbc
