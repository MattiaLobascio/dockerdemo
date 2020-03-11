from flask import Flask, request, render_template
import numpy as np
import string
import re
from functools import reduce
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = stopwords.words('english')
stop_words = set(stop_words)


def paragraph2sentences(text):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    text = re.sub("\s+", " ", text.replace(".", ". ").replace(". . . ", "... ")).strip()
    return sent_detector.tokenize(text)

def preprocess_sentences(sentence_list):
    ps = PorterStemmer ()
    lemmatizer = WordNetLemmatizer()
    words = [re.sub("[^a-z\s]+", "", s.lower()).split(" ") for s in sentence_list]
    words = [[lemmatizer.lemmatize(ps.stem(w)) for w in sentence] for sentence in words]
    return [list(set(w) - stop_words - set([''])) for w in words]

def weighed_freq_occurrence(tokenized_sentence_list):
    aux_list = [word for sentence in tokenized_sentence_list for word in sentence]
    aux_list = sorted(aux_list, key=lambda x: aux_list.count(x), reverse=True)
    max_freq = aux_list.count(aux_list[0])
    return {word:aux_list.count(word)/max_freq for word in aux_list}

def sentence_weighed_freq(tokenized_sentence_list, weights):
    aux_list = [[weights[w] if w in weights.keys() else 0 for w in s] for s in tokenized_sentence_list]
    return [sum(s) for s in aux_list]

def sort_and_select(sentence_list, sentence_weights, n_sentences):
    aux_list = list(zip(sentence_list, sentence_weights))
    aux_list = sorted(aux_list, key=lambda x: x[1], reverse=True)
    aux_list = [sentence for sentence,weight in aux_list]
    
    n_sentences = abs(n_sentences)
    if n_sentences<1:
        n_sentences = np.ceil(len(aux_list)*n_sentences)
    if n_sentences>len(aux_list) or n_sentences==0:
        n_sentences = len(aux_list)
    
    return " ".join(aux_list[:int(n_sentences)])

def nltk_summarizer_n(text, n=0.3):
    sentence_list = paragraph2sentences(text)
    tokenized_sentence_list = preprocess_sentences(sentence_list)
    word_weights = weighed_freq_occurrence(tokenized_sentence_list)
    sentence_weights = sentence_weighed_freq(tokenized_sentence_list, word_weights)
    return sort_and_select(sentence_list, sentence_weights, n)





app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')
    
@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    output = nltk_summarizer_n(text)
    return render_template('form.html', final=output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True, threaded=True)
