import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist, pos_tag
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Read the Moby Dick file
with open('moby_dick.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenization
tokens = word_tokenize(text.lower())

# Stop-words filtering
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token not in stop_words]

# Parts-of-Speech (POS) tagging
pos_tags = pos_tag(filtered_tokens)

# POS frequency
pos_counts = FreqDist(tag for (word, tag) in pos_tags)
top_pos = pos_counts.most_common(5)

print("Top 5 most common parts of speech:")
for pos, count in top_pos:
    print(f"{pos}: {count}")

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word, pos=pos[0].lower()) for word, pos in pos_tags[:20]]

print("\nLemmatized tokens:")
print(lemmatized_tokens)

# Plotting frequency distribution
pos_counts.plot(30, cumulative=False)
plt.title('Frequency Distribution of POS')
plt.xlabel('Parts of Speech')
plt.ylabel('Frequency')
plt.show()