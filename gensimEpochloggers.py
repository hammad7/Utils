#####
##	Usage
#####

# epoch_logger = EpochLogger()
# epoch_saver = EpochSaver(get_tmpfile("/tmp/temporary_model"),3)##SAve model after every 3 iters

# model = Doc2Vec(window=10, size = 500, alpha=0.025, dm=0, dbow_words=1,
#                 min_alpha=0.025, epochs=40, seed= 123,
#                 min_count = 5, report_delay=1, workers=3,callbacks=[epoch_logger,epoch_saver] )


import time
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import get_tmpfile

class EpochLogger(CallbackAny2Vec):
    "Callback to log information about training"
    def __init__(self,test,testData):
        self.epoch = 1
        self.test = test
        self.modelSeed = 2
        
        # test set
        self.testData = testData


    def on_epoch_begin(self, model):
        self.time1 = time.time() 
        print("Epoch #{} start".format(self.epoch))
    
    def on_epoch_end(self, model):
        self.time2 = time.time() 
        print("Epoch #{} end".format(self.epoch)+"  time(s):"+str(int(self.time2-self.time1)))
        self.epoch += 1
        if self.test:## get copy not in testmodel since takes memory and this is debugging feature
          ###### getCopy() does the trick!!! 
        	results = testModel(getCopy(model),self.testData["testIds"],self.testData["tagged_sentence_list"],self.modelSeed)
            # del modelcopy #=getCopy(model)
	        print (results)


class EpochSaver(CallbackAny2Vec):
    "Callback to save model after every epoch"
    def __init__(self, path_prefix,skip_epochs=3):
        self.path_prefix = path_prefix
        self.epoch = 0
        self.skip_epochs = skip_epochs
    def on_epoch_end(self, model):
        if self.epoch%self.skip_epochs==0:
            output_path = '{}_epoch{}.model'.format(self.path_prefix, self.epoch)
            print("Save model to {}".format(output_path))
            model.save(output_path)
        self.epoch += 1



########
### helpers
########
import time
import random
import collections
from os.path import isfile, join
from os import listdir
from gensim.models.doc2vec import TaggedDocument
import numpy as np
from tqdm import tqdm
from copy import copy, deepcopy

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print ('%r took %2.2f s' % (method.__name__, (te - ts)))
        return result
    return timed

def getFilePath(directory,prefix, fileExt = ".csv", timeDiff = 3600, sep="_"):
    # if prefix=="doc2vec":
    #     return  "/media/mohd/New Volume/git_repo/AutoAnswering/gensim_doc2vec/QuestionRecommendation/src/core/training/models/autoAnsweringSavedModel_6apr/autoAnsweringModel_6Apr"
    if prefix=="dataDump":
        return  "/media/mohd/New Volume/git_repo/AutoAnswering/gensim_doc2vec/QuestionRecommendation/src/core/training/data/dataDump_1540913417.csv"
    if not directory or not prefix:
        return ""
    fileList = []
    for f in listdir(directory):
        absolutePath = join(directory, f)
        if isfile(absolutePath) and f.startswith(prefix) and not f.endswith(".npy"):
            fileList.append(absolutePath)
    fileList.sort(reverse=True)
    if not len(fileList)>0:
        return ""
    if timeDiff:
        lastTime = int(fileList[0].split(prefix+sep)[1].replace(fileExt,""))
        currentTime = int(time.time())
        if currentTime - lastTime <= timeDiff:
            return fileList[0]
    return ""


def prepareData(dataDump,testOnEachEpoch,testSet,testSetSeed):
    print(len(dataDump),"sssssssss",dataDump[0])
    tagged_sentence_list = [] 
    for single_row in dataDump: 
        # if(len(single_row) > 10):#################
        question_id = single_row[0]#######################3
        # print(single_row)
        question = cleanDoc(single_row[1])######################
        tokens = question.split()
        if(len(tokens) >= 3):
            single_tagd_doc = TaggedDocument(words = tokens, tags =[str(question_id)])#############
            tagged_sentence_list.append(single_tagd_doc)

    testIds = []
    if testOnEachEpoch:
        random.seed(testSetSeed)
        print("Using:",testSet)
        # testSetData = random.sample(tagged_sentence_list[testSet["index"][0]:testSet["index"][1]], testSet["docs"])# initial test set for comparison of results
        # testIndice = [i for i,row in enumerate(testSetData)]
        testIndice = random.sample(range(testSet["index"][0],testSet["index"][1]), testSet["docs"])# initial test set for comparison of results
    return {"testIds":testIndice, "tagged_sentence_list":tagged_sentence_list}

def testModel(model,testIds,tagged_sentence_list,modelSeed):
    ranks = []
    print(len(tagged_sentence_list),modelSeed,"sssssssss")
    print(len(testIds),"sssssssss")
    for doc_id in tqdm(testIds):
        if modelSeed:
            model.random.seed(modelSeed)
        # print(tagged_sentence_list[doc_id].words)
        inferred_vector = model.infer_vector(tagged_sentence_list[doc_id].words,alpha=0.025,min_alpha=0.025,steps=5) ## global tagged_sentence_list
        ques_id = tagged_sentence_list[doc_id].tags[0]
        sims = model.docvecs.most_similar([inferred_vector], topn=1000+1)## because top 1000
        array_of_docids = []
        for docid, sim in sims:
            try:
                # array_of_docids.append(int(docid.strip("'")))
                array_of_docids.append(docid)
            except Exception as e:
                print("####### string doc ids: ",docid)
        # array_of_docids = [int(docid.strip("'")) for docid, sim in sims]
        # array_of_docids = [docid for docid, sim in sims]
        # print(array_of_docids,ques_id)
        rank=1001
        # print(type(ques_id),type(array_of_docids[0]))
        # print((ques_id),(array_of_docids[0]))
        if ques_id in array_of_docids:
            rank = array_of_docids.index(ques_id)+1
        ranks.append(rank)
    meanRank = str(np.mean([row for row in ranks if row>0]))
    hits=collections.Counter(ranks)
    print(hits)
    sum1=0
    sum10=0
    sum50=0
    sum100=0
    sum500=0
    sum1000=0
    for key,cnt in dict(hits).items():
        if key!=-1:
            if key==1:
                sum1+=(cnt)
            if key<=10:
                sum10+=(cnt)
            if key<=50:
                sum50+=(cnt)
            if key<=100:
                sum100+=(cnt)
            if key<=500:
                sum500+=(cnt)
            if key<=1000:
                sum1000+=(cnt)
    return (meanRank,float(sum1)/len(ranks),float(sum10)/len(ranks),float(sum50)/len(ranks),float(sum100)/len(ranks),float(sum500)/len(ranks),float(sum1000)/len(ranks))


def cleanDoc(doc):
    return doc.lower().replace(",", " , ").replace("...", " . ").replace("??", " ? ").replace("?", " ? ").replace("(", " ( ").replace(")", " ) ").replace("&amp;", "")


class Copyable:
    __slots__ = 'a', '__dict__'
    def __init__(self, a):
        self.a = a
    def __copy__(self):
        return type(self)(self.a)
    def __deepcopy__(self, memo): # memo is a dict of id's to copies
        id_self = id(self)        # memoization avoids unnecesary recursion
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                deepcopy(self.a, memo))
            memo[id_self] = _copy 
        return _copy

def getCopy(model):
    c1 = Copyable(model)
    c2 = deepcopy(c1)
    return c2.a

