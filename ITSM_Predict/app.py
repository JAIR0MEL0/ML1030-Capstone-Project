from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.externals import joblib
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__)
acc = 0

@app.route('/',methods=['GET','POST'])
def index():
        errors = []
        results = {}

        if request.method == 'POST':
                try:
                        desc = request.form['incident']
                        #incident = requests.get(desc)
                        incident = [desc]
                except:
                        errors.append("Can't read the field")
                        return render_template('index.html', errors=errors)
                if incident:
                        
                       #load Category model
                        clf = joblib.load('model-simple_cat.pkl')
                        labels = pd.read_csv('labels_cat.csv', header=None, index_col=0, skiprows=1, names = ['id', 'name'])
                        tokenizer = Tokenizer()
                        tokenizer.fit_on_texts(incident)
                        post_seq = tokenizer.texts_to_sequences(incident)
                        post_seq_padded = pad_sequences(post_seq, maxlen=500)
                        pred_cat = clf.predict(post_seq_padded)
                        arregle = labels.iloc[[np.argmax(pred_cat)],:1]
                        pred_mcat = arregle.name.item()
                        p_mcat = np.amax(pred_cat)
                        
                        #load Sub Category model
                        clf = joblib.load('model-simple_sub.pkl')
                        labels = pd.read_csv('labels_sub.csv', header=None, index_col=0, skiprows=1, names = ['id', 'name'])
                        tokenizer = Tokenizer()
                        tokenizer.fit_on_texts(incident)
                        post_seq = tokenizer.texts_to_sequences(incident)
                        post_seq_padded = pad_sequences(post_seq, maxlen=500)
                        pred_cat = clf.predict(post_seq_padded)
                        arregle = labels.iloc[[np.argmax(pred_cat)],:1]
                        pred_sub = arregle.name.item()
                        p_sub = np.amax(pred_cat)

                        #load Priority model
                        clf = joblib.load('model-simple_prt.pkl')
                        labels = pd.read_csv('labels_prt.csv', header=None, index_col=0, skiprows=1, names = ['id', 'name'])
                        tokenizer = Tokenizer()
                        tokenizer.fit_on_texts(incident)
                        post_seq = tokenizer.texts_to_sequences(incident)
                        post_seq_padded = pad_sequences(post_seq, maxlen=500)
                        pred_cat = clf.predict(post_seq_padded)
                        arregle = labels.iloc[[np.argmax(pred_cat)],:1]
                        pred_prt = arregle.name.item()
                        p_prt = np.amax(pred_cat)

                        #load Support Group model
                        clf = joblib.load('model-simple_grp.pkl')
                        labels = pd.read_csv('labels_grp.csv', header=None, index_col=0, skiprows=1, names = ['id', 'name'])
                        tokenizer = Tokenizer()
                        tokenizer.fit_on_texts(incident)
                        post_seq = tokenizer.texts_to_sequences(incident)
                        post_seq_padded = pad_sequences(post_seq, maxlen=500)
                        pred_cat = clf.predict(post_seq_padded)
                        arregle = labels.iloc[[np.argmax(pred_cat)],:1]
                        pred_grp = arregle.name.item()
                        p_grp = np.amax(pred_cat)


                        #load Resolution Category model
                        clf = joblib.load('model-simple_sol.pkl')
                        labels = pd.read_csv('labels_sol.csv', header=None, index_col=0, skiprows=1, names = ['id', 'name'])
                        tokenizer = Tokenizer()
                        tokenizer.fit_on_texts(incident)
                        post_seq = tokenizer.texts_to_sequences(incident)
                        post_seq_padded = pad_sequences(post_seq, maxlen=500)
                        pred_cat = clf.predict(post_seq_padded)
                        arregle = labels.iloc[[np.argmax(pred_cat)],:1]
                        pred_sol = arregle.name.item()
                        p_sol = np.amax(pred_cat)


                        #load Knowledge Article model
                        clf = joblib.load('model-simple_KCS.pkl')
                        labels = pd.read_csv('labels_KCS.csv', header=None, index_col=0, skiprows=1, names = ['id', 'name'])
                        tokenizer = Tokenizer()
                        tokenizer.fit_on_texts(incident)
                        post_seq = tokenizer.texts_to_sequences(incident)
                        post_seq_padded = pad_sequences(post_seq, maxlen=500)
                        pred_cat = clf.predict(post_seq_padded)
                        arregle = labels.iloc[[np.argmax(pred_cat)],:1]
                        pred_KCS = arregle.name.item()
                        p_KCS = np.amax(pred_cat)


                        
                        results = [('Category', pred_mcat, p_mcat),
                                   ('Sub Category', pred_sub, p_sub),
                                   ('Priority', pred_prt, p_prt),
                                   ('Support Group', pred_grp, p_grp),
                                   ('Resolution Category', pred_sol, p_sol),
                                   ('Potential Knowledge Article', pred_KCS, p_KCS)]
        return render_template('index.html', errors = errors, results=results)

if __name__ == '__main__':
	app.run(debug=True)
