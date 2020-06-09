import pickle
from os import path

import django
from django.conf import settings
from django.db.models import Q
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .settings import DATABASES, INSTALLED_APPS

if not settings.configured:
    settings.configure(DATABASES=DATABASES, INSTALLED_APPS=INSTALLED_APPS)
    django.setup()
    from sentiment_tool.sentiment_tool.models import *


class ItalianSentimentAnalyzer:
    DATASET_SVM_PICKLE = '../dataset/svm.pickle'

    @classmethod
    def train(cls):
        documents = TextPattern.objects.filter(~Q(sentiment=None)).all()
        pipeline_svm = Pipeline([
            ('feature_vect', TfidfVectorizer(strip_accents='unicode',
                                             tokenizer=word_tokenize,
                                             stop_words=stopwords.words('italian'),
                                             decode_error='ignore',
                                             analyzer='word',
                                             norm='l2',
                                             ngram_range=(1, 3)
                                             )),
            ('clf', SVC(probability=True,
                        C=1,
                        shrinking=True,
                        kernel='rbf'))
        ])
        x_data = [doc.text for doc in documents]
        y_data = [doc.sentiment for doc in documents]
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data,
            test_size=0.2,
            # random_state=42
        )
        pipeline_svm.fit(x_train, y_train)
        y_pred = pipeline_svm.predict(x_test)
        print(accuracy_score(y_test, y_pred))
        print(precision_recall_fscore_support(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        pickle.dump(pipeline_svm, open(cls.DATASET_SVM_PICKLE, "wb"))

    def predict(self, text):
        if not path.exists(self.DATASET_SVM_PICKLE):
            self.train()
        pipeline_svm = pickle.load(open(self.DATASET_SVM_PICKLE, "rb"))
        return {
            'sentiment': pipeline_svm.predict([text])[0],
            'probability': max(pipeline_svm.predict_proba([text])[0]),
        }


if __name__ == '__main__':
    clf = ItalianSentimentAnalyzer()
    clf.train()
    print(clf.predict('Il rumore di martello e trapano di un operaio al lavoro la mattina 7,30'))
    print(clf.predict('Camera spaziosa, letti largho e comodi, colazione varia e personale gentile'))
    print(clf.predict(
        "L'hotel avrebbe bisogno di una ripulita, non tanto le stanze dato che la mia era spaziosa e confortevole, ma il corridoio e la moquette."))
