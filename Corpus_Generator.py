import pymongo
from pymongo import MongoClient
import nltk
from nltk.corpus import stopwords
import json
import string
import codecs
import gensim 
from gensim import corpora, models
from collections import Counter
import re
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import nltk.stem

class GensimCorpus(object):
    def __init__(self,corpus_text_file,diction):
        self.corpus_text_file = corpus_text_file
        self.dictionary = diction
        
    def __iter__(self):
        for line in open(self.corpus_text_file):
            yield self.dictionary.doc2bow(line.split())

def dictionary_generator(corpus_file):
    dictionary = corpora.Dictionary(line.split() for line in open(corpus_file))
    return dictionary

def create_models(corpus_file):
    dictionary = dictionary_generator(corpus_file)
    print('Created Dictionary')
    corp = GensimCorpus(corpus_file,dictionary)
    print('Created Corpus Object')
    tfidf = models.TfidfModel(corp)
    print('Created TFIDF Model')
    tfidf_corp = tfidf[corp]
    print('Created TFIDF Corpus')
    return dictionary,corp,tfidf,tfidf_corp

def load_models(dictionary_file,corpus_file,tfidf_file):
    dictionary = corpora.Dictionary.load(dictionary_file)
    corp = GensimCorpus(corpus_file,dictionary)
    tfidf = models.TfidfModel.load(tfidf_file)
    tfidf_corp = tfidf[corp]
    return dictionary,corp,tfidf,tfidf_corp

def tfidf_filtered_corpus_generator(corpus_filename,threshold):
    corpus_filename = 'tfidf_filtered_'+str(threshold).strip('.')+'.txt'
    ind=0
    with codecs.open(corpus_filename,'a',encoding='utf8') as f:
        for doc in tfidf_corp:
            if ind%500000 == 0:
                print(ind)
            f.write(' '.join([dictionary[i] for i,j in doc if j>=threshold]))
            f.write('\n')
            ind+=1

def remove_unicode_punct(subj, chars):
    return re.sub(u'(?u)[' + re.escape(''.join(chars)) + ']', ' ', subj)
                
def create_cranberry_corpus(file_name,sanitizer):
    ind = 0 
    with codecs.open(file_name,'a',encoding='utf8') as f:
        for rec in ch.find({'crossref_doi':True}):
            ex = sanitizer(rec['title'])+u'\n'
            f.write(export)
            ind+=1
            if ind%100000==0:
                print(ind)
    
def create_raspberry_corpus(file_name,sanitizer):
    ind = 0 
    with codecs.open(file_name,'a',encoding='utf8') as f:
        for rec in coops.find({'abstract': {'$exists': True}, '$where': "this.abstract.length>0"}):
            san_title = sanitizer(rec['title'])
            san_abs = sanitizer(rec['abstract'])
            f.write(san_title+' '+san_abs+'\n')
            ind+=1
            if ind%10000==0:
                print(ind)

####Filterers

def stop_word_sanitise(title,stops):
    #lower case, strip whitespace and carriages, remove stopwords, remove punctuation
    lt = title.lower()
    slt = lt.strip()
    tslt = remove_unicode_punct(slt,punct_filter)
    stop_filtered = [i for i in tslt.split() if i not in stops]
    export = u' '.join(stop_filtered)
    return export

def minimal_sanitise(title):
    #lower case, strip whitespace and carriages, remove punctuation
    lt = title.lower()
    slt = lt.strip()
    tslt = remove_unicode_punct(slt,punct_filter)
    export = tslt.strip()
    return export

def stemming_sanitise(title,stops,stemmer):
    #lower case, strip whitespace and carriages, remove punctuation, remove stopwords, stem
    lt = title.lower()
    slt = lt.strip()
    tslt = remove_unicode_punct(slt,punct_filter)
    stop_filtered = [i for i in tslt.split() if i not in stops]
    stem_filtered = [stemmer.stem(i) for i in stop_filtered]
    export = u' '.join(stem_filtered)
    return export
    
def lemmatizing_sanitise(title,stops,lemmatizer):
    #lower case, strip whitespace and carriages, remove punctuation, remove stopwords, lemmatize
    lt = title.lower()
    slt = lt.strip()
    tslt = remove_unicode_punct(slt,punct_filter)
    stop_filtered = [i for i in tslt.split() if i not in stops]
    lemma_filtered = [lemmatizer.lemmatize(words) for words in stop_filtered]
    export = u' '.join(lemma_filtered)
    return export

def get_corpus_stats(in_file,diction,outfile_name):
    unique_word_count=0
    for k in diction.iterkeys():
        if unique_word_count<k:
            unique_word_count=k
    print('Counted Unique Words')
    word_freq = Counter()
    word_count = 0
    document_count = 0
    document_lengths = Counter()
    interim_corp = GensimCorpus(in_file,diction)
    ind=0
    for doc in interim_corp:
        word_count+=len(doc)
        document_count+=1
        document_lengths.update([len(doc)])
        upd = []
        for w_id,w_freq in doc:
            upd+=([w_id]*w_freq)
        word_freq.update(upd)
        ind+=1
        if ind%10000==0:
            sys.stdout.write('\r[{0}] {1}'.format('#'*(ind/10000), ind))
            sys.stdout.flush()
    mean_doc_length = float(word_count)/float(document_count)
    mode_doc_length = document_lengths.most_common(1)[0]
    print('\nGenerated Counting Stats')
    ranked_word_freq = word_freq.most_common()
    ziphian_table = []
    for rank in range(unique_word_count):
        w = dictionary[ranked_word_freq[rank][0]].encode('utf-8')
        r = rank+1
        log_r = math.log(r,10)
        f = ranked_word_freq[rank][1]
        log_f = math.log(f,10)
        ziphian_table.append((w,r,log_r,f,log_f))
    print('Generated Ziphian Data')
    sample = []
    log_ranks = list(zip(*ziphian_table)[2])
    for s in np.arange(ziphian_table[0][2],ziphian_table[-1][2],ziphian_table[1][2]):   
        sample.append(log_ranks.index(min(log_ranks,key=lambda x:abs(x-s))))
    z_grad,z_c = np.polyfit([ziphian_table[s][2] for s in sample],[ziphian_table[s][4] for s in sample],1)
    line_freq = map(lambda x: z_grad*x+z_c,log_ranks)
    plt.close()
    plt.plot(log_ranks,list(zip(*ziphian_table)[4]),'r')
    plt.plot(log_ranks,line_freq,'b')
    plt.savefig(outfile_name+'_ziphian_plot.png')
    print('Saved Ziphian Plot')
    plt.close()
    plt.plot(document_lengths.keys(),document_lengths.values())
    plt.savefig(outfile_name+'_document_word_lengths.png')
    plt.close()
    print('Saved Document Length Distribution Plot')
    with open(outfile_name+'_ziphian_data.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['word','rank','log rank','freqency','log_frequncy'])
        writer.writerows(ziphian_table)
    print('Writen Ziphian Data To file')
    with open(outfile_name+'_document_lengths.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['words per document','number of documents'])
        writer.writerows(zip(document_lengths.keys(),document_lengths.values()))
    print('Writen Document Length Distribution data to file')
    with open(outfile_name+'stats.txt','wb') as f:
        f.write('Word count : '+str(word_count)+'\n')
        f.write('Unique words : ' + str(unique_word_count)+'\n')
        f.write('Mean document word count : ' + str(mean_doc_length)+'\n')
        f.write('Mode document word count : '+ str(mode_doc_length[0])+'\n')
        f.write('Document count : ' + str(document_count)+'\n')
        f.write('Ziphian gradient : '+str(z_grad)+'\n')
        f.write('Ziphian intercept : '+str(z_c)+'\n')
        f.write('most_frequent 10 words : '+'\n')
        for w in ziphian_table[0:10]:
            f.write('"'+w[0]+'" : '+str(w[3])+' occurances\n')
    print('Written stats report to file')

def compare_stemmers(dictionary,outfile_name):
    lancaster = nltk.stem.lancaster.LancasterStemmer()
    porter = nltk.stem.porter.PorterStemmer()
    snowball = nltk.stem.snowball.EnglishStemmer()
    wordnet = nltk.stem.WordNetLemmatizer()
    it = dictionary.iteritems()
    lancaster_reps=0
    porter_reps=0
    snowball_reps=0
    wordnet_reps=0
    lancaster_porter_agreements = 0
    lancaster_snowball_agreements = 0
    lancaster_wordnet_agreements = 0
    porter_snowball_agreements = 0
    porter_wordnet_agreements = 0
    snowball_wordnet_agreements=0
    ind=0
    with open(outfile_name, 'wb') as f:
        writer = csv.writer(f, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['word','lancaster','porter','snowball','wordnet'])
        for w_id,w in it:
            wl = lancaster.stem(w)
            wp = porter.stem(w)
            ws = snowball.stem(w)
            ww = wordnet.lemmatize(w)
            wl_fire = w!=wl
            wp_fire = w!=wp
            ws_fire = w!=ws
            ww_fire = w!=ww
            lancaster_reps+=int(wl_fire)
            porter_reps+=int(wp_fire)
            snowball_reps+=int(ws_fire)
            wordnet_reps+=int(ww_fire)
            if (wl_fire|wp_fire|ws_fire|ww_fire):
                row = [w.encode('utf-8'),'','','','']
                if wl_fire: row[1]=wl.encode('utf-8')
                if wp_fire: row[2]=wp.encode('utf-8')
                if ws_fire: row[3]=ws.encode('utf-8')
                if ww_fire: row[4]=ww.encode('utf-8')
                writer.writerow(row)
                lancaster_porter_agreements +=int(wl==wp)
                lancaster_snowball_agreements +=int(wl==ws)
                lancaster_wordnet_agreements +=int(wl==ww)
                porter_snowball_agreements +=int(wp==ws)
                porter_wordnet_agreements +=int(wp==ww)
                snowball_wordnet_agreements +=int(ws==ww)
            if ind%10000==0:
                sys.stdout.write('\r[{0}] {1}'.format('#'*(ind/10000), ind))
                sys.stdout.flush()
            ind+=1
    print('finished')
    print('lancaster replacements: '+str(lancaster_reps))
    print('porter replacements: '+str(porter_reps))
    print('snowball replacements: '+str(snowball_reps))
    print('wordnet replacements: '+str(wordnet_reps))
    print('lancaster_porter_agreements: ' + str(lancaster_porter_agreements))
    print('lancaster_snowball_agreements: ' +str(lancaster_snowball_agreements))
    print('lancaster_wordnet_agreements: '+str(lancaster_wordnet_agreements))
    print('porter_snowball_agreements: '+str(porter_snowball_agreements))
    print('porter_wordnet_agreements: '+str(porter_wordnet_agreements))
    print('snowball_wordnet_agreements: '+str(snowball_wordnet_agreements))

##Connect to Database
#mongo_url = 'mongodb://localhost:6666/'
mongo_url = 'mongodb://localhost:27017/'
db = 'Cherry'
coll_in = 'Cranberry'
client = MongoClient(mongo_url)
ch = client[db][coll_in]
coops = client[db]['raspberry']

##prepare punction filters, stopword lists,stemming objects
punct_filter = [
    u'"',u'#',u'$',u'%',u'&',u'\\',u"'",u'(',u')',u'*',u'+',u',',u'.',u'/',
    u'-',u':',u';',u'<',u'=',u'>',u'?',u'@',u'[',u']',u'^',u'_',u'`',u'{',
    u'|',u'}',u'–',u'\u2013',u'\u2010',u'\u2606',u'\u22C5',u'\u201D',
    u'\u2248',u'\u21CC',u'\u223C',u'\u2212',u'\u2014',u'\u2032',u'\u2018',
    u'\u2019',u'\u2022',u'\u2020',u'\u00B0',u'\u201C',u'\u29B9',u'\uFF0D',
    u'\u2261'
    ]
stop = stopwords.words('english')
with open('chemistry_stopwords.json') as f:
    chem_stop = json.load(f)
max_stop = stop+chem_stop
stemmer = nltk.stem.snowball.EnglishStemmer()

##create the corpus text file
create_raspberry_corpus('stemmed_raspberry_corpus.txt',lambda x: stemming_sanitise(x,max_stop,stemmer))

##create the models associated wth the text file and save them to disk in case required
dictionary, corpus, tfidf_model,tfidf_corpus = create_models('stemmed_raspberry_corpus.txt')
dictionary.save('stemmed_raspberry_dictionary')
tfidf_model.save('stemmed_raspberry_tfidf_model')
tfidf_corpus.save('stemmed_raspberry_tfidf_corpus')

##load the models associated with a corpus
#dictionary,corp,tfidf,tfidf_corp = load_models(
#    'raw_raspberry_dictionary',
#    'raw_raspberry_corpus.txt',
#    'raw_raspberry_tfidf_model'
#)

##run corpus diagnostics
get_corpus_stats('stemmed_raspberry_corpus.txt',dictionary,'stemmed_raspberry')



