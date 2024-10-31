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

## Model Training

- **Train-Test Split:** Data is divided into training and test sets with a standard 80-20 split.
- **Modeling:**
  - Three classifiers—K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest—are trained on the dataset. Hyperparameters for each model are kept default initially but can be adjusted for optimization.
  - Each model is fitted to the training data, and predictions are generated for the test data.
- **Evaluation:**
  - Precision, recall, and F1-score are computed for each model’s predictions, allowing for performance comparison.
 
```python
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, Y_train)
Y_pred = knn_model.predict(X_test)
precision = precision_score(Y_test, Y_pred, average='weighted')
```

## Cross-Validation

- A 10-fold cross-validation is conducted to evaluate model consistency. The function NFoldVal is called for each model, providing metrics for each fold and the overall mean and standard deviation.
- Cross-validation results reveal how each model performs on different subsets, highlighting robustness and stability.

```python
NFoldVal(knn_model, 10, X_train, Y_train)
```

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
