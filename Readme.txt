Sentiment Analysis on Product Reviews
What is this project about?
This project is all about teaching a computer to understand the feelings in product reviews.
The computer reads a review and then decides if the review is:

Positive (happy or good review)

Negative (unhappy or bad review)

Neutral (neither good nor bad)

Why is this important?
When lots of people write reviews about products online, it’s hard for businesses to read and understand all of them.
This project helps automatically find out what people feel about a product — saving time and helping businesses improve.

How does it work? Step-by-step
1. Collect Data
We have many product reviews labeled as positive, negative, or neutral. This labeled data helps the computer learn.

2. Clean the Text
Reviews often have unnecessary words like "the", "and", or "is" that don’t help in understanding the feeling.
We remove these common words (called stopwords) and also clean the text by making everything lowercase and removing punctuation.

3. Convert Text to Numbers (Feature Extraction)
Computers can’t understand words directly, so we convert the cleaned reviews into numbers using TF-IDF Vectorization.
TF-IDF gives higher importance to words that are unique or important in a review and lowers the weight of common words.

4. Split Data
We split our dataset into two parts:

Training set: The computer learns from this data.

Testing set: We check how well the computer learned by testing it on new reviews it hasn’t seen before.

5. Train Machine Learning Models
We use two models:

Logistic Regression: A simple, fast model that works well for text classification.

Random Forest: A collection of decision trees that vote on the best answer, improving accuracy.

6. Evaluate the Models
We check how well the models are performing by measuring:

Accuracy: How many reviews were correctly classified.

Precision & Recall: How good the model is at finding each type of sentiment.

F1-score: A balance between precision and recall.

7. Visualize Results
We also create a confusion matrix to see where the model makes mistakes — for example, mixing up neutral and positive reviews.