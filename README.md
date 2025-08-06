# Mystery_Postcard_Classifier
This project uses a simple Naive Bayes classifier to determine the author of a "mystery postcard" by comparing its text to the writing samples of three historical figures: Emma Goldman, Matthew Henson, and Wu Tingfang.

How It Works
    The script combines the writings of the three individuals, treats each as a separate category, and then trains a machine learning model to recognize their distinct writing styles.       This is a classic example of text classification.

The core logic involves these steps:

    Data Preparation: The writings of Goldman, Henson, and Wu are loaded and labeled.
    Vectorization: The text is converted into numerical feature vectors using a Bag-of-Words model (CountVectorizer). This model counts the occurrences of each word in the text.
    Training: A MultinomialNB (Multinomial Naive Bayes) classifier is trained on the feature vectors and their corresponding labels.
    Prediction: The trained classifier is then used to predict the author of the new, unseen text from the mystery postcard.

Getting Started
Prerequisites
    Make sure you have Python 3 installed on your system. You will also need the scikit-learn library.
    You can install scikit-learn using pip:
    pip install -U scikit-learn

File Structure
    Your project should be organized with the following files:
    main.py: The main script that runs the classification.
    goldman_emma_raw.py: A Python file containing a list of writing samples from Emma Goldman.
    henson_matthew_raw.py: A Python file containing a list of writing samples from Matthew Henson.
    wu_tingfang_raw.py: A Python file containing a list of writing samples from Wu Tingfang.

Running the Script
    To run the classifier, simply execute the main.py script from your terminal:

    python main.py

The script will output the predicted author of the mystery postcard.

Customization
    You can easily customize this project to use different texts or authors.
    
    Create new _raw.py files with lists of text samples for your new authors.
    
    Import them into main.py.
    
    Update the friends_docs and friends_labels lists to include your new data.
    
    Change the final if/elif/else block to return the names of your new authors.
