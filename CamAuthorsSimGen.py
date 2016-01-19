import gensim
import pymongo
from pymongo import MongoClient
import time

mongo_url = 'mongodb://localhost:27017/'
db = 'CamSim'
coll = 'CamAuthors'
client = MongoClient(mongo_url)
ca = client[db][coll]

sim_name = 'SimEngine'
model = gensim.models.Word2Vec.load(sim_name)
y = time.time()
ind = 0
cur2 = ca.find()
huge_object = []
for rec3 in cur2:
    name2 = rec3['name'] 
    corp2 = rec3['corpus'].split()
    huge_object.append((name2,corp2))
cur1=ca.find({'sims':{'$exists':0}},no_cursor_timeout=True)
for rec1 in cur1:
    corp1 = rec1['corpus']
    sims = []
    x = time.time()
    for n,c in huge_object:
        s = model.n_similarity(corp1.split(),c)
        sims.append({n:s})
    ca.update_one({'name':rec1['name']},{'$set':{'sims':sims}})
    print ('Record ' + unicode(str(ind))+ ': ' + unicode(str(time.time() - x)) + u' seconds to produce sims for '+rec1['name'])
    ind+=1
print('******************Finished**********************')
print(str(ind) + 'records generated in '+ str(time.time() - y) + ' seconds')
