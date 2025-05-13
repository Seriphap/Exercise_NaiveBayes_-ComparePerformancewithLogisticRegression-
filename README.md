# Exercise: NaiveBayes (compare performance with LogisticRegression)
<p><font size="4">&nbsp;&nbsp;&nbsp;View in Steamlit (Web Application): Waiting...! </p> 
<br>

## Step1: Import Data
<p><font size="4">&nbsp;&nbsp;&nbsp;Dataset: https://drive.google.com/file/d/1nd-aBohJM2BsbDRwdpNQqss1HtcT8kSc/view?usp=drive_link </p>

```python
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/MyData/emails.csv')
```
<img src="https://github.com/user-attachments/assets/bff91138-eb8e-4cf9-9cfc-7aa26a2c15dc" width="40%">

## Step2: Apply NaiveBayes for training and testing model

```python
# Split the data into train and test sets (80% train, 20% test)
vectorizer = CountVectorizer()# Create an instance of CountVectorizer
X = vectorizer.fit_transform(df['text'])
indices = np.arange(len(df))
X_train_NB, X_test_NB, y_train_NB, y_test_NB, train_indices_NB, test_indices_NB = train_test_split(X, y, indices,test_size=0.2, random_state=42)
```
```python
#Create a Naive Bayes classification model
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train_NB, y_train_NB)  # train 80%
```
```python
# Make predictions on the train set
y_predict_train_NB = NB_classifier.predict(X_train_NB)
print("Classification Report for Training Set:")
print('Predicted train values:', y_predict_train_NB)
print(classification_report(y_train_NB, y_predict_train_NB))
accuracy_train_NB = accuracy_score(y_train_NB, y_predict_train_NB)
print(f'Accuracy: {accuracy_train_NB * 100:.2f}%')
print("Confusion Matrix:")
print(confusion_matrix(y_train_NB, y_predict_train_NB))
```
```python
# Make predictions on the trest set
y_predict_test_NB = NB_classifier.predict(X_test_NB) # test 20%
print("\nClassification Report for Testing Set:")
print('Predicted test values:', y_predict_test_NB)
print(classification_report(y_test_NB, y_predict_test_NB))
accuracy_test_NB = accuracy_score(y_test_NB, y_predict_test_NB)
print(f'Accuracy: {accuracy_test_NB * 100:.2f}%')
print("Confusion Matrix:")
print(confusion_matrix(y_test_NB, y_predict_test_NB))
```
## Step3: Apply Logistic Regression for training and testing model
```python
# Split the data into train and test sets (80% train, 20% test)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)  # Converts the text to a sparse matrix
indices = np.arange(len(y))
X_train_LG, X_test_LG, y_train_LG, y_test_LG, train_indices_LG, test_indices_LG = train_test_split(X, y, indices, test_size=0.2, random_state=42)
```
```python
#Create a a Logistic Regression model
model = LogisticRegression(max_iter=1000)
```
```python
# Make predictions on the train set
y_predict_train_LG = model.predict(X_train_LG)
print("Classification Report for Training Set:")
print('Predicted train values:', y_predict_train_LG)
print("Classification Report:")
print(classification_report(y_train_LG, y_predict_train_LG))
accuracy_train_LG = accuracy_score(y_train_LG, y_predict_train_LG)
print(f'Accuracy: {accuracy_train_LG * 100:.2f}%')
print("Confusion Matrix:")
print(confusion_matrix(y_train_LG, y_predict_train_LG))
```

```python
# Make predictions on the test set
y_predict_test_LG = model.predict(X_test_LG)
print("\nClassification Report for test dataset:")
print('Predicted test values:', y_predict_test_LG)
print("Classification Report:")
print(classification_report(y_test_LG, y_predict_test_LG))
accuracy_test_LG = accuracy_score(y_test_LG, y_predict_test_LG)
print(f'Accuracy: {accuracy_test_LG * 100:.2f}%')
print("Confusion Matrix:")
print(confusion_matrix(y_test_LG, y_predict_test_LG))
```
## Step4: Evaluate the model's performance

| Naive Bayes | Logistic Regression |
|--------|---------|
| <img src="https://github.com/user-attachments/assets/d8c78e19-b179-46d6-a8d1-70c0d3ca4c52" width="400"> | <img src="https://github.com/user-attachments/assets/722c9557-a77d-4d8f-9ce7-50cc602e8695" width="400">|

<img src="https://github.com/user-attachments/assets/1ea9adcb-b1f4-49b7-92f0-39f6e82be896" width="90%">










