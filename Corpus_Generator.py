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

class GensimCorpus(object):
'''Memory friendly class for a GenSim bag of words corpus
from a text file with documents seperated line by line'''
    def __init__(self,corpus_text_file,diction):
        self.corpus_text_file = corpus_text_file
        self.dictionary = diction
        
    def __iter__(self):
        for line in open(self.corpus_text_file):
            yield self.dictionary.doc2bow(line.split())

def dictionary_generator(corpus_file):
'''Create a gensim dictionary from a corpus text file with 
documents seperated line by line'''
    dictionary = corpora.Dictionary(line.split() for line in open(corpus_file))
    return dictionary

def create_models(corpus_file):
'''Shortcut method to quickly generate a set of gensim models for a corpus text file'''
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
'''Shortcut method to quickly load in a set of gensim models on disk'''
    dictionary = corpora.Dictionary.load(dictionary_file)
    corp = GensimCorpus(corpus_file,dictionary)
    tfidf = models.TfidfModel.load(tfidf_file)
    tfidf_corp = tfidf[corp]
    return dictionary,corp,tfidf,tfidf_corp

def tfidf_filtered_corpus_generator(corpus_filename,threshold):
'''generate a text corpus on disk that includes words with a tfidf score
above a threshold value '''
    corpus_filename = 'tfidf_filtered_'+str(threshold).strip('.')+'.txt'
    ind=0
    with codecs.open(corpus_filename,'a',encoding='utf8') as f:
        for doc in tfidf_corp:
            if ind%500000 == 0:
                print(ind)
            f.write(' '.join([dictionary[i] for i,j in doc if j>=threshold]))
            f.write('\n')
            ind+=1

def raw_corpus_generator(file_name):
'''generate a text corpus without filtering stopwords. 
Punctuation still removed. Only article titles'''
    ind = 0 
    with codecs.open(file_name,'a',encoding='utf8') as f:
        for rec in ch.find({'crossref_doi':True}):
            lt = rec['title'].lower()
            slt = lt.strip()
            tslt = slt.translate(punct_filter)
            export = tslt+u'\n'
            f.write(export)
            ind+=1
            if ind%100000==0:
                print(ind)

def remove_unicode_punct(subj, chars):
'''remove unicode punctuation characters from sentence'''
    return re.sub(u'(?u)[' + re.escape(''.join(chars)) + ']', ' ', subj)
                
def sanitise(title):
'''filtering method for raw text scraped online.
casts all to lowercase, strips trailing newlines and whitespace,
removes punctuation and removes stopwords'''
    lt = title.lower()
    slt = lt.strip()
    tslt = remove_unicode_punct(slt,punct_filter)
    stop_filtered = [i for i in tslt.split() if i not in stop]
    export = u' '.join(stop_filtered)
    return export

def create_stopword_filtered_corpus(file_name):
'''generate a text corpus with stopwords and
punctuation removed, only article titles'''
    ind = 0 
    with codecs.open(file_name,'a',encoding='utf8') as f:
        for rec in ch.find({'crossref_doi':True}):
            f.write(sanitise(rec['title'])+'u\n')
            ind+=1
            if ind%10000==0:
                print(ind)
                
def create_stopword_filtered_raspberry_corpus(file_name):
'''generate a text corpus with stopwords and
punctuation removed, both titles and abstracts'''
    ind = 0 
    with codecs.open(file_name,'a',encoding='utf8') as f:
        for rec in coops.find({'abstract': {'$exists': True}, '$where': "this.abstract.length>0"}):
            san_title = sanitise(rec['title'])
            san_abs = sanitise(rec['abstract'])
            f.write(san_title+' '+san_abs+'\n')
            ind+=1
            if ind%10000==0:
                print(ind)

def get_corpus_stats(in_file,diction,outfile_name):
'''method that generates statistics about a text corpus
with documents seperated by newlines. Creates statistics on disk,
prepended with argument "outfile_name"'''
    #count unique words
    unique_word_count=0
    for k in diction.iterkeys():
        if unique_word_count<k:
            unique_word_count=k
    print('Counted Unique Words')
    
    #count word frequencies,document lengths, wordcounts,document counts
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
        if ind%10000==0: #print an indicator of progress
            sys.stdout.write('\r[{0}] {1}'.format('#'*(ind/10000), ind))
            sys.stdout.flush()
    #calc mean and mode document word counts
    mean_doc_length = float(word_count)/float(document_count)
    mode_doc_length = document_lengths.most_common(1)[0]
    print('Generated Counting Stats')
    #sort word frequencies into ascending order
    ranked_word_freq = word_freq.most_common()
    #generate a table of word ranks,frequencies and log ranks,log frequencies
    ziphian_table = []
    for rank in range(unique_word_count):
        w = dictionary[ranked_word_freq[rank][0]].encode('utf-8')
        r = rank+1
        log_r = math.log(r,10)
        f = ranked_word_freq[rank][1]
        log_f = math.log(f,10)
        ziphian_table.append((w,r,log_r,f,log_f))
    print('\nGenerated Ziphian Data')
    #generate straight line fit for ziphian
    sample = []
    log_ranks = list(zip(*ziphian_table)[2])
    for s in np.arange(ziphian_table[0][2],ziphian_table[-1][2],ziphian_table[1][2]):   
        sample.append(log_ranks.index(min(log_ranks,key=lambda x:abs(x-s))))
    z_grad,z_c = np.polyfit([ziphian_table[s][2] for s in sample],[ziphian_table[s][4] for s in sample],1)
    line_freq = map(lambda x: z_grad*x+z_c,log_ranks)
    #plot ziphian distribution
    plt.close()
    plt.plot(log_ranks,list(zip(*ziphian_table)[4]),'r')
    plt.plot(log_ranks,line_freq,'b')
    plt.savefig(outfile_name+'_ziphian_plot.png')
    print('Saved Ziphian Plot')
    plt.close()
    #plot document lengths 'histogram'
    plt.plot(document_lengths.keys(),document_lengths.values())
    plt.savefig(outfile_name+'document_word_lengths.png')
    plt.close()
    print('Saved Document Length Distribution Plot')
    #write out the data
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

#connect to databases
#mongo_url = 'mongodb://localhost:6666/'
mongo_url = 'mongodb://localhost:27017/'
db = 'Cherry'
coll_in = 'Cranberry'
client = MongoClient(mongo_url)
ch = client[db][coll_in]
coops = client[db]['raspberry']

punct_filter = [u'"',u'#',u'$',u'%',u'&',u'\\',
u"'",u'(',u')',u'*',u'+',u',',u'.',u'/',u'-',
u':',u';',u'<',u'=',u'>',u'?',u'@',u'[',u']',
u'^',u'_',u'`',u'{',u'|',u'}',u'–',u'\u2013',
u'\u2010',u'\u2606',u'\u201D',u'\u2248',
u'\u223C',u'\u2212',u'\u2014',u'\u2032',
u'\u2018',u'\u2019',u'\u2022',u'\u2020',
u'\u00B0',u'\u29B9',u'\uFF0D',u'\u2261'
]
stop = stopwords.words('english')

corpus_textfile = 'second_raspberry_corpus.txt'
out_stats_name = 'RASPBERRY'
#create all models and save to disk
create_stopword_filtered_raspberry_corpus(corpus_textfile)
dictionary, corpus, tfidf_model,tfidf_corpus = create_models(corpus_textfile)
dictionary.save(corpus_textfile.split('.')[0]+'_dictionary')
tfidf_model.save(corpus_textfile.split('.')[0]+'_tfidf_model')
tfidf_corpus.save(corpus_textfile.split('.')[0]+'_tfidf_corpus')
#load all models
#dictionary,corp,tfidf,tfidf_corp = load_models('second_raspberry_dictionary','second_raspberry_corpus.txt','second_raspberry_tfidf_model')
get_corpus_stats(corpus_textfile,dictionary,out_stats_name)
