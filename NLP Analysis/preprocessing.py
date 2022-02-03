import nltk.data
from sklearn.model_selection import train_test_split
from pathlib import Path
from bnlp import NLTKTokenizer

def sentence_split_english(raw_text):
    nltk.download('punkt')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return tokenizer.tokenize(raw_text)   

def sentence_split_bengali(raw_text):
    bnltk = NLTKTokenizer()
    sentence_tokens = bnltk.sentence_tokenize(raw_text)
    st = []
    for i in sentence_tokens:
        st.append(i.split('br'))
    flat_list = [item for sublist in st for item in sublist]    
    return flat_list

def preprocess(sentence):
    punctuations = '''!()-[]{};:"\,<>./?@#$%^&'*_~”\n“–=`'''
    lstring = sentence.lower().strip()
    for p in lstring:
        if p in punctuations:
            lstring = lstring.replace(p, "")
    
    # remove all double whitespaces
    lstring = " ".join(lstring.split())
    return lstring

def data_preparation_english(file_path):
    raw_text = ""
    with open(file_path, 'r', encoding='utf-8') as fhand:
        for line in fhand:
            # if it's not empty line
            if line:
                raw_text += line.strip() + " "
    sentences = sentence_split_english(raw_text)
    for i in range(len(sentences)):
        sentences[i] = preprocess(sentences[i])

    train, test = train_test_split(sentences, test_size=0.2, random_state=1)

    Path("english/corpus").mkdir(parents=True, exist_ok=True)

    with open('english/corpus/train.txt', 'w') as fhand:
        for sentence in train:
            fhand.write("%s\n" % sentence)

    with open('english/corpus/test.txt', 'w') as fhand:
        for sentence in test:
            fhand.write("%s\n" % sentence)
    # print('\n-----\n'.join(test))

def data_preparation_bengali(file_path):
    raw_text = ""
    with open(file_path, 'r', encoding='utf-8') as fhand:
        for line in fhand:
            # if it's not empty line
            if line:
                raw_text += line.strip() + " "
    sentences = sentence_split_bengali(raw_text)
    for i in range(len(sentences)):
        sentences[i] = preprocess(sentences[i])

    train, test = train_test_split(sentences, test_size=0.2, random_state=1)
    Path("bengali/corpus").mkdir(parents=True, exist_ok=True)

    with open('bengali/corpus/train.txt', 'w') as fhand:
        for sentence in train:
            if sentence:
                fhand.write("%s\n" % sentence)

    with open('bengali/corpus/test.txt', 'w') as fhand:
        for sentence in test:
            if sentence:
                fhand.write("%s\n" % sentence)       
