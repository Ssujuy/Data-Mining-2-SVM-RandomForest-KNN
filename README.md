# Data-Mining-2-SVM-RandomForest-KNN
This notebook applies data mining techniques for sentiment analysis using K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest models. Precision, recall, and F1-score are calculated through 10-fold cross-validation, providing insight into model reliability and classification accuracy.

## Helper Functions

- Splits data into N folds and iterates over each, training the model on the training folds and evaluating it on the validation fold.
- Computes metrics like precision, recall, and F1-score for each fold, then averages these scores across folds to provide mean values and standard deviations.

```python
def NFoldVal(model, N, X, Y):
    kf = KFold(n_splits=N, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        model.fit(X_train_fold, Y_train_fold)
        # Predictions and metrics computation
```

## Preprocessing

- **Import Prerequisites:** Libraries are loaded for data handling (pandas), NLP tasks (nltk), model evaluation, and embedding (transformers, sklearn). Warnings are suppressed for cleaner output.

- **Text Cleaning and Tokenization:**
  - Stopwords are removed, punctuation is filtered, and text is tokenized. The cleaned text is vectorized using TF-IDF for numerical input to models.
  - GloVe embeddings provide dense vector representations, enhancing model accuracy by capturing semantic relationships.
 
```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text_column'])
sequences = tokenizer.texts_to_sequences(data['text_column'])
word_index = tokenizer.word_index
```

## Annotate data with pretrained model

- Initializes a sentiment analysis pipeline and selects the 'comments' column from the sampled DataFrame for annotation.
- Annotates each comment, truncates it to 512 characters, and stores the sentiment label in a new column 'sentiment'.

```python
annotator = pipeline("sentiment-analysis")

comments_to_annotate = year2019_sample['comments']

annotations = []

for comment in comments_to_annotate:
    comment = comment[:512]
    annotation = annotator(comment)
    annotations.append(annotation)

year2019_sample.loc[:1999, 'sentiment'] = [annotation[0]['label'] for annotation in annotations]

print(year2019_sample.head(10))
```

## Sentiment Distribution

- Groups the sampled DataFrame by 'sentiment', counts occurrences, and resets the index to create a summary DataFrame.
- Plots a bar chart to visualize the distribution of sentiments, adding labels to each bar and displaying the plot.

```python
temp = year2019_sample.groupby('sentiment').count().reset_index()

plt.figure(figsize=(6, 6))
plt.bar(temp['sentiment'], temp['comments'], color='skyblue')
plt.xlabel('Month')
plt.ylabel('Average Price')
plt.title('Sentiment distribution in 2019')

for i, price in enumerate(temp['comments']):
    plt.text(i, price + 0.5, f'{price:.2f}', ha='center', va='bottom')

plt.ylim(bottom=0)
plt.tight_layout()
plt.show()
```

## Model Training

- **Train-Test Split:** Data is divided into training and test sets with a standard 80-20 split.
- **TF-IDF**
  - **Vectorization**
    - Initializes a TF-IDF vectorizer with English stopwords and transforms the training data into TF-IDF vectors.
    - Transforms the test data too into TF-IDF vectors and prints the shape of the resulting training and test matrices.


```python
vectorizer = TfidfVectorizer(stop_words='english')

X_train_vectors_2019 = vectorizer.fit_transform(X_train_2019)

X_test_vectors_2019 = vectorizer.transform(X_test_2019)

print("X train vector shape: " , X_train_vectors_2019.shape)
print("Y train vector shape: " , X_test_vectors_2019.shape)
```

- **SVM**
    - Trains an SVM model with a linear kernel on the TF-IDF vectorized training data and makes predictions on the test data.
    - Calculates and prints precision, recall, and F1-score, along with a detailed classification report, and performs 10-fold cross-validation.

```python
y_pred_svm_2019 = svm_model_2019.predict(tfidf_df_test_2019)

precision = precision_score(Y_test_2019, y_pred_svm_2019, average='weighted')
recall = recall_score(Y_test_2019, y_pred_svm_2019, average='weighted')
f1 = f1_score(Y_test_2019, y_pred_svm_2019, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

print(classification_report(Y_test_2019, y_pred_svm_2019))
```

- **Random forest**
    - Trains a Random Forest classifier with 100 trees on the TF-IDF vectorized training data and makes predictions on the test data.
    - Calculates and prints precision, recall, and F1-score, along with a detailed classification report, and performs 10-fold cross-validation.

```python
y_pred_rf_2019 = rf_model_2019.predict(X_test_vectors_2019)

precision = precision_score(Y_test_2019, y_pred_rf_2019, average='weighted')
recall = recall_score(Y_test_2019, y_pred_rf_2019, average='weighted')
f1 = f1_score(Y_test_2019, y_pred_rf_2019, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

report_rf_2019 = classification_report(Y_test_2019, y_pred_rf_2019)
print("Random Forest Classification Report:")
print(report_rf_2019)
```

- **KNN**
    - Trains a K-Nearest Neighbors (KNN) classifier with 5 neighbors on the TF-IDF vectorized training data and makes predictions on the test data.
    - Calculates and prints precision, recall, and F1-score, along with a detailed classification report, and performs 10-fold cross-validation.

```python
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, Y_train)
Y_pred = knn_model.predict(X_test)
precision = precision_score(Y_test, Y_pred, average='weighted')
```

- **Cross-Validation**

    - A 10-fold cross-validation is conducted to evaluate model consistency. The function NFoldVal is called for each model, providing metrics for each fold and the overall mean and standard deviation.
    - Cross-validation results reveal how each model performs on different subsets, highlighting robustness and stability.

```python
NFoldVal(knn_model, 10, X_train, Y_train)
```

## Word Embeddings

- **Vectorization**
    - Compute Average Embedding
    - Loads GloVe embeddings from a file into a dictionary and initializes a tokenizer to convert text data into sequences of integers.
    - Pads these sequences to a fixed length and computes average embeddings for each padded sequence using the embedding matrix.

## Results Analysis

- After cross-validation, results for each model are compared. The metrics are averaged, with each model’s strengths analyzed based on precision, recall, and F1-score stability:
  - KNN: Generally effective but sensitive to data distribution; its performance may vary more than the other models.
  - SVM: Shows strong generalization and performs well on precision, although computationally more intensive.
  - Random Forest: Offers robust performance, achieving a balance of precision, recall, and F1-score across folds, indicating strong stability.
- Results are visualized to compare models, illustrating metric variance and enabling quick insights into model consistency.

## Conclusion

- Based on metric averages and standard deviations from cross-validation, the Random Forest model emerges as the most reliable choice for this dataset due to its balanced performance across folds.
- KNN showed promise with high accuracy in certain folds, but its performance variability suggests that parameter tuning may be necessary for production use.
- SVM displayed consistent performance in terms of precision, suggesting that it may be a suitable choice for applications where minimizing false positives is critical.
