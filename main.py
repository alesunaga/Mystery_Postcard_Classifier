
# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Raw text data from friends
from goldman_emma_raw import goldman_docs
from henson_matthew_raw import henson_docs
from wu_tingfang_raw import wu_docs

def identify_mystery_friend(mystery_postcard):
    """
    Identifies the author of a mystery postcard using a Naive Bayes classifier.

    Args:
        mystery_postcard (str): The text of the mystery postcard.

    Returns:
        str: The name of the friend who likely wrote the postcard.
    """
    # Combine all friends' writing samples into a single list
    friends_docs = goldman_docs + henson_docs + wu_docs
    
    # Create corresponding labels for each friend
    # 1: Emma Goldman, 2: Matthew Henson, 3: Wu Tingfang
    friends_labels = [1] * len(goldman_docs) + [2] * len(henson_docs) + [3] * len(wu_docs)

    # Initialize the CountVectorizer
    bow_vectorizer = CountVectorizer()
    
    # Fit the vectorizer to the friends' documents and transform them into feature vectors
    friends_vectors = bow_vectorizer.fit_transform(friends_docs)
    
    # Transform the mystery postcard text into a feature vector
    mystery_vector = bow_vectorizer.transform([mystery_postcard])

    # Initialize the Multinomial Naive Bayes classifier
    friends_classifier = MultinomialNB()
    
    # Train the classifier with the friends' feature vectors and labels
    friends_classifier.fit(friends_vectors, friends_labels)
    
    # Predict the author of the mystery postcard
    predictions = friends_classifier.predict(mystery_vector)
    
    # Determine the friend's name based on the prediction
    if predictions[0] == 1:
        return "Emma Goldman"
    elif predictions[0] == 2:
        return "Matthew Henson"
    elif predictions[0] == 3:
        return "Wu Tingfang"
    else:
        return "someone else"

# Text from the mystery postcard
mystery_postcard = """
Discussing the activities and rôle of the Anarchists in the Revolution,
Kropotkin said: "We Anarchists have talked much of revolutions, but
few of us have been prepared for the actual work to be done during the
process. I have indicated some things in this relation in my 'Conquest
of Bread.' Pouget and Pataud have also sketched a line of action in
their work on 'How to Accomplish the Social Revolution.'" Kropotkin
thought that the Anarchists had not given sufficient consideration
to the fundamental elements of the social revolution. The real facts
in a revolutionary process do not consist so much in the actual
fighting--that is, merely the destructive phase necessary to clear
the way for constructive effort. The basic factor in a revolution is
the organization of the economic life of the country. The Russian
Revolution had proved conclusively that we must prepare thoroughly for
that. Everything else is of minor importance. He had come to think that
syndicalism was likely to furnish what Russia most lacked: the channel
through which the industrial and economic reconstruction of the country
may flow. He referred to Anarcho-syndicalism. That and the coöperatives
would save other countries some of the blunders and suffering Russia
was going through.
"""

# Identify the author and print the result
mystery_friend = identify_mystery_friend(mystery_postcard)
print(f"The postcard was from {mystery_friend}!")
