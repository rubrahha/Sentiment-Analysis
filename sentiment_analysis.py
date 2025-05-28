import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ===========================================
# STEP 1: Create a larger dataset (90 reviews)
# ===========================================
positive_reviews = [
    "Amazing product, exceeded my expectations!",
    "Excellent service and great quality.",
    "I'm very happy with this purchase.",
    "Totally worth the price!",
    "Best item I’ve bought in a while.",
    "Superb design and smooth finish.",
    "Very satisfied with this.",
    "It works perfectly and looks great.",
    "Outstanding performance and value.",
    "I love it, will buy again.",
    "This made my day, thank you!",
    "Highly recommend this to everyone.",
    "It does exactly what it promises.",
    "Five stars from me!",
    "Perfect in every way.",
    "Great build quality and fast delivery.",
    "Absolutely fantastic product.",
    "It’s a great deal for the price.",
    "Can’t believe how good it is.",
    "Very happy customer here.",
    "My favorite item now.",
    "Works like a charm.",
    "Pleasantly surprised by the quality.",
    "Great customer experience.",
    "I’m impressed, honestly.",
    "No complaints at all.",
    "Everything I wanted and more.",
    "Smooth and easy to use.",
    "Incredible value!",
    "Just wow, love it!"
]

negative_reviews = [
    "Terrible experience, do not buy.",
    "Waste of money, very disappointed.",
    "Item broke after one day.",
    "Poor quality and awful service.",
    "Totally not worth the price.",
    "Regret this purchase.",
    "Disappointed and frustrated.",
    "Stopped working almost instantly.",
    "The worst product I’ve ever used.",
    "Save your money, avoid this.",
    "Extremely poor performance.",
    "Packaging was damaged and used.",
    "Feels cheap and unreliable.",
    "Nothing like what was advertised.",
    "Very bad service experience.",
    "It was broken on arrival.",
    "I’m returning this ASAP.",
    "Unusable and useless.",
    "Complete letdown.",
    "Terrible quality control.",
    "Unacceptable and frustrating.",
    "Doesn’t work as expected.",
    "Would give zero stars if I could.",
    "Total waste.",
    "Horrible delivery and product.",
    "Felt cheated.",
    "Bitter experience.",
    "This is a scam.",
    "Threw it in the trash.",
    "Doesn’t even deserve a review."
]

neutral_reviews = [
    "It's okay, nothing special.",
    "Average product, does the job.",
    "Not good, not bad, just neutral.",
    "Works fine, but not impressive.",
    "Meh, it's alright I guess.",
    "Exactly as expected.",
    "Decent for the price.",
    "Meets basic expectations.",
    "Nothing extraordinary here.",
    "It's usable, nothing more.",
    "Fine, but wouldn’t buy again.",
    "Just a regular item.",
    "Serves its purpose.",
    "No strong opinion about this.",
    "It works as it should.",
    "Does what it says.",
    "Kind of plain, but acceptable.",
    "Middle of the road.",
    "Neither great nor terrible.",
    "Mediocre, but functional.",
    "Can’t complain, can’t praise.",
    "Simple and standard.",
    "Mildly satisfied.",
    "Nothing exciting.",
    "Not a memorable purchase.",
    "Very basic.",
    "Reasonable quality for the price.",
    "Neutral experience.",
    "Expected this result.",
    "It’s just okay."
]

# Combine all into one DataFrame
reviews = positive_reviews + negative_reviews + neutral_reviews
sentiments = ['positive'] * 30 + ['negative'] * 30 + ['neutral'] * 30

df = pd.DataFrame({'Review': reviews, 'Sentiment': sentiments})

# ===========================================
# STEP 2: Clean the text
# ===========================================
def preprocess(text):
    words = text.lower().split()
    cleaned_words = [word for word in words if word.isalpha() and word not in stop_words]
    return " ".join(cleaned_words)

df['Cleaned_Review'] = df['Review'].apply(preprocess)

# ===========================================
# STEP 3: TF-IDF Vectorization
# ===========================================
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['Cleaned_Review'])  # input features
y = df['Sentiment']                            # output labels

# ===========================================
# STEP 4: Split into Train/Test sets
# ===========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# ===========================================
# STEP 5: Train the models
# ===========================================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

# ===========================================
# STEP 6: Show the results
# ===========================================
print("\n🔍 Logistic Regression Results:")
print(classification_report(y_test, pred_lr, zero_division=0))

print("🔍 Random Forest Results:")
print(classification_report(y_test, pred_rf, zero_division=0))

# ===========================================
# STEP 7: Visualize confusion matrix
# ===========================================
cm = confusion_matrix(y_test, pred_rf, labels=['positive', 'neutral', 'negative'])

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['positive', 'neutral', 'negative'],
            yticklabels=['positive', 'neutral', 'negative'])

plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
