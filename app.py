from flask import Flask,request,render_template
import re
import nltk
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
emoticon_pattern = re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')


app = Flask(__name__)

with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)


def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = emoticon_pattern.findall(text)
    text = re.sub('[\W+]', ' ', text.lower()) + ' '.join(emojis).replace('-', '')
    prter = PorterStemmer()
    text = [prter.stem(word) for word in text.split() if word not in stopwords_set]

    return " ".join(text)


@app.route('/')
def index():
    return render_template(index.html)

@app.route('/predict',methods=['POST','Get'])
def predict():
    if request.method == 'POST':
        comment = request.form['text']
        cleaned_comment = preprocessing(comment)
        comment_vecto=tfidf.transform([cleaned_comment])
        prediction=clf.predict(comment_vecto)[0]
        return render_template('index.html',prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)

