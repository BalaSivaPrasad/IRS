# Install scikit-learn (if not already)
!pip install -q scikit-learn

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

# Load 4 categories from 20 Newsgroups dataset for simplicity
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5)
X = vectorizer.fit_transform(newsgroups.data)
y_true = newsgroups.target

# Apply K-Means clustering
k = len(categories)
model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=10, random_state=42)
model.fit(X)
y_pred = model.labels_

# --------- Purity Calculation ----------
def purity_score(y_true, y_pred):
    contingency = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

# --------- Evaluation Metric ----------
print("Purity Score:", purity_score(y_true, y_pred))
