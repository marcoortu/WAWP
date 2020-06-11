import pickle
from os import path

import django
import numpy as np
from django.conf import settings
from django.db.models import Q
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split

if not settings.configured:
    from sentiment_tool.sentiment_tool.settings import DATABASES, INSTALLED_APPS

    settings.configure(DATABASES=DATABASES, INSTALLED_APPS=INSTALLED_APPS)
    django.setup()
    from sentiment_tool.sentiment_tool.models import *
else:
    from .models import *


class ItalianSentimentAnalyzer:
    SVM_PICKLE_PATH = '{}/dataset/svm.pickle'.format(settings.BASE_DIR)
    VECTORIZER_PICKLE_PATH = '{}/dataset/tfidf_vect.pickle'.format(settings.BASE_DIR)
    CALIBRATOR_PICKLE_PATH = '{}/dataset/calibrator.pickle'.format(settings.BASE_DIR)

    @classmethod
    def get_dataset(cls):
        documents = TextPattern.objects.filter(~Q(sentiment=None)).all()
        feature_vect = TfidfVectorizer(
            strip_accents='unicode',
            tokenizer=word_tokenize,
            stop_words=stopwords.words('italian'),
            decode_error='ignore',
            analyzer='word',
            norm='l2',
            ngram_range=(1, 3)
        )
        x_data = feature_vect.fit_transform([doc.text for doc in documents])
        y_data = [doc.sentiment for doc in documents]
        pickle.dump(feature_vect, open(cls.VECTORIZER_PICKLE_PATH, "wb"))
        return x_data, y_data

    @classmethod
    def train(cls):
        """
            SGDClassifier (Stochastic Gradient Descent ) implements a Linear SVM
            with:
            loss = 'hinge' (https://en.wikipedia.org/wiki/Hinge_loss)
            penalty = 'L2' (Eucliedean distance, root square of sum of squares of vector elements)
            https://scikit-learn.org/stable/modules/sgd.html#sgd
        :return: None
        """
        x_data, y_data = cls.get_dataset()
        clf = SGDClassifier(
            loss="hinge",
            penalty="l2",
            max_iter=20
        )
        clf.partial_fit(x_data, y_data, classes=np.unique(y_data))
        calibrator = CalibratedClassifierCV(clf, cv='prefit')
        model = calibrator.fit(x_data, y_data)
        pickle.dump(clf, open(cls.SVM_PICKLE_PATH, "wb"))
        pickle.dump(model, open(cls.CALIBRATOR_PICKLE_PATH, "wb"))

    def predict(self, text):
        if not path.exists(self.SVM_PICKLE_PATH):
            self.train()
        clf = pickle.load(open(self.SVM_PICKLE_PATH, "rb"))
        feature_vectorizer = pickle.load(open(self.VECTORIZER_PICKLE_PATH, "rb"))
        calibrator = pickle.load(open(self.CALIBRATOR_PICKLE_PATH, "rb"))
        feature_vect = feature_vectorizer.transform([text])
        return {
            'sentiment': clf.predict(feature_vect)[0],
            'probability': max(calibrator.predict_proba(feature_vect)[0]),
        }


if __name__ == '__main__':
    ItalianSentimentAnalyzer.train()
    clf = SGDClassifier(
        loss="hinge",
        penalty="l2",
        max_iter=20
    )
    x_data, y_data = ItalianSentimentAnalyzer.get_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.2,
    )
    clf.partial_fit(x_train, y_train, classes=np.unique(y_train))
    y_pred = clf.predict(x_test)
    print(accuracy_score(y_test, y_pred))
    print(precision_recall_fscore_support(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    ItalianSentimentAnalyzer.train()
    clf = ItalianSentimentAnalyzer()
    print(clf.predict('Il rumore di martello e trapano di un operaio al lavoro la mattina 7,30'))
    print(clf.predict('Camera spaziosa, letti largho e comodi, colazione varia e personale gentile'))
    print(clf.predict("""
    L'hotel avrebbe bisogno di una ripulita, non tanto le stanze dato che la mia era spaziosa e confortevole, 
    ma il corridoio e la moquette.
    """))
