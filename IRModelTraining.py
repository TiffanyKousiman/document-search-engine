import glob, os, re
import util
import math

####################################################### TF-IDR IR Model #############################################################

def tfidf(doc, df, ndocs):
    
    """ 
    calculate tf*idf value (weight) of every term in a BowDoc object
    
    Params
    -------
    doc (BowDoc)            : a BowDoc object
    df (dict)               : a {term:df, ...} dictionary
    ndocs                   : number of documents in a given BowColl collection

    Returns
    -------
    {term:tfidf_weight , ...} dictionary for the given BowDoc object
    """

    # initialize a dict to store tfidf value for each term in the document 
    tfidf_ = {}

    # retrieve the term-frequency dictionary using `terms` attribute of `doc`
    tf_ = doc.get_term_freq_dict()

    # calculate the normalisation factor
    normFactor = 0
    for term, tf in tf_.items():
        normFactor += ((math.log(tf, 10) + 1)*(math.log(ndocs/df[term], 10)))**2
    normFactor = math.sqrt(normFactor)

    # loop through each term and compute its tfidf score
    for term, tf in tf_.items():
        tfidf = ((math.log(tf, 10) + 1)*(math.log(ndocs/df[term], 10)))/normFactor
        tfidf_[term] = tfidf        

    return tfidf_


def tfidf_IR(q, docColl):

    """
    calculate document relevance score for a given query and a document using tf*idf-based document and query
    feature functions

    Params
    -------
    q (dict) : term-frequency dictionary for query terms
    docColl (BowColl Collection) : a collection of DocWord to calculate the document weight for 

    Returns
    -------
    {docid:docWeight, ...} dictionary containing document score for each document for a given query
    """ 

    # get document freq dictionary and collection size from docColl
    df = docColl.get_doc_freq()
    ndocs = docColl.get_num_docs()

    # initialise the returned dictionary of {docid: docWeight}
    R = {} 

    for docid, docWord in docColl.get_docs().items():

        # initialise R 
        R[docid] = 0
        
        # create tfidf dict with respect to this docword
        tfidf_dict = tfidf(docWord, df, ndocs)

        # for term exist in both document and query, compute its document x query term weight 
        # and add it to the document relevance score
        for term, tfidf_ in tfidf_dict.items():
            if (term in q):
                R[docid] += tfidf_*q[term]
    
    R = {k: v for k, v in sorted(R.items(), key=lambda item: item[1], reverse=True)}
    
    return R

####################################################### Baseline: BM25 Model #############################################################

def bm25(coll, q):

    """ 
    calculate documents BM25 score for all documents in collection coll given 
    query q.
    
    Params
    -------
    coll (BowColl)  : collection of docWord objects in {docid:docWord,...} dictionary
    q (dict)        : query terms and frequency in {queryTerm:frequency,...} dictionary

    Returns
    -------
    {docID: bm25_score,...} for all documents in the collection coll
    """
    
    # set parameters
    k1 = 1.2
    k2 = 100
    b = 0.75
    avdl = docColl.get_avg_doc_len()    # calculate average doc length for collection `coll`
    N = coll.get_num_docs()           # N = number of documents in the collection
    R = 0                               # R = number of relevant documents in the collection
    
    # initialise an empty dictionary to store bm25 scores
    bm25_score = {}
    
    # get doc frequency
    df = docColl.get_doc_freq() 

    for docid, docWord in coll.get_docs().items():

        # initialise R 
        bm25_score[docid] = 0

        dl = docWord.get_doc_len()  # dl = document length of docWord
        K = k1*((1-b) + b * (dl/avdl)) # calculate K 

        for term, tf in docWord.get_term_freq_dict().items():
            if (term in q):
                n = df[term]    # n = document frequency of `term`
                r = 0           # r = relevant document frequency of `term` - assumed to be 0 given R = 0
                f = tf          # f = term frequency in document 
                qf = q[term]    # qf = term frequency in query 
                
                # calculate bm25 score for this document 
                bm25_score[docid] += (math.log((((r + 0.5)/(R - r + 0.5))/((n - r + 0.5)/(N - n - R + r + 0.5))),2) *
                                    (((k1 + 1) * f)/(K + f)) * (((k2 + 1) * qf)/(k2 + qf)))
    
    bm25_score = {k: v for k, v in sorted(
            bm25_score.items(), key=lambda item: item[1], reverse=True)
    }

    return bm25_score


####################################################### My_Model1: Rocchio Model #############################################################

def rocchio(query_term_freq, vocabList, Rel, NonRel, docColl):

    """
    Calculate the new query terms using Rocchio Query Expansion

    Params
    -------
    query_term_freq (dict)  : term-frequency dictionary for query terms
    vocabList (list)        : a union set of terms in the relevant document set
    Rel (list)              : a list of ids of relevant documents
    NonRel (list)           : a list of ids of non-relevant documents
    docColl (BowColl)       : document collection

    Returns
    -------
    {query term : query weight, ...} dictionary containing new weights for the top 50 query terms
    """

    nonrel_weight = {}
    rel_weights = {}
    new_query_term_freq_dict = {}

    len_Rel = len(Rel)
    len_NonRel = len(NonRel)
    df = docColl.get_doc_freq()
    ndocs = docColl.get_num_docs()

    alpha = 8
    beta = 16
    gamma = 4

    for vocab in vocabList:

        # initialise the relevance and nonrelevance weights
        rel_weights[vocab] = 0
        nonrel_weight[vocab] = 0
        
        # get the tfidf weighting of each query term 
        for docid, doc in docColl.get_docs().items():
            tfidf_ = tfidf(doc, df, ndocs)
            for term, weight in tfidf_.items():
                if term == vocab:
                    if docid in Rel:
                        rel_weights[vocab] += weight
                    else:
                        nonrel_weight[vocab] += weight
        
        # find the new query weight 
        if vocab not in query_term_freq.items():
            old_w = 0
        else:
            old_w = query_term_freq.items()[vocab]
            
        new_w = alpha*old_w + beta*(1/len_Rel)*rel_weights[vocab] - gamma*(1/len_NonRel)*nonrel_weight[vocab]

        try: 
            new_query_term_freq_dict[vocab] += new_w
        except KeyError:
            new_query_term_freq_dict[vocab] = new_w
    
    # top 50 query terms
    new_query_term_freq_dict = dict(sorted(\
        new_query_term_freq_dict.items(), key=lambda item: item[1], reverse=True)[:50])

    return new_query_term_freq_dict


####################################################### My_Model2: Pseudo-Relevance Model #############################################################

def query_likelihood(query, docColl):
    
    """
    Estimate the P(Q|D) for all docs in docColl usin the Dirichlet Smoothing Query Likelihood model and
    return for each document in the collection a score

    Params
    -------
    query                   : a list of query terms
    docColl (BowColl)       : document collection

    Returns
    -------
    {docid: score, ...} sorted dictionary containing likelihood score for each document
    """

    ql_score = {}                                   # query-likelihood document score
    total_coll_len = docColl.get_total_doc_len()    # total number of word occurences in the collection
    cum_tf = docColl.get_cum_freq()                 # cummulative term frequency of terms in the collection {term:cum_tf, ...}
    mu = 1500                                       # parameter 

    # estimate P(Q|D) for all docs in docColl
    for docid, doc in docColl.get_docs().items():  
        query_score = 0                             # initialise query likelihood to be 0
        doc_len = doc.get_doc_len()                 # document length
        for term in query:  
            # get the term frequency of term in doc
            try:
                tf = doc.get_term_freq_dict()[term]
            except KeyError:
                tf = 0      
            # calculate the score
            try:
                query_score += math.log((tf + mu * (cum_tf[term] / total_coll_len)) / (doc_len + mu))
            except KeyError:  
                query_score += 0                    # if term is neither in document or collection
        ql_score[docid] = query_score       # add query-likelihood-score to dictionary
    
    # sort dictionary by setting in the descending relevance
    query_likelihood_sort = {key: value for key, value in
                sorted(ql_score.items(), key=lambda item: item[1], reverse=True)}
    
    return query_likelihood_sort  

def relevance_model_prob(ql_score, query_term_list, docColl, k):
    """
    Estimate the relevance model probabilities for all terms in the collection (cum_tf)

    Params
    -------
    ql_score                : output from query_likelihood()
    query_term_list         : list of query terms for a document collection
    docColl                 : document collection
    k                       : used for getting the top-k documents with the highest query_likelihood score 

    Returns
    -------
    {term: score, ...} sorted dictionary containing the weighted average of the language model probabilities 
    """
    term_likelihood = {} 
    relevant_docs = dict(list(ql_score.items())[:k])        # get top k {docid:ql_score, ...} 
    cum_tf = docColl.get_cum_freq()                         # cummulative term frequency of terms in the collection {term:cum_tf, ...}

    for term in cum_tf:
        prob_w_r = 0                                        # prob_w_r is relevance model probability 
        for docid in relevant_docs:                         
            doc = docColl.get_doc(docid)
            doc_len = doc.get_doc_len()
            # 1. calculate p(Q|D)
            prob_Q_D = 0
            for q_term in query_term_list:
                try:
                    q_tf = doc.get_term_freq_dict()[q_term]  # term frequency of query term q_term in doc
                except KeyError:
                    q_tf = 0
                prob_Q_D += q_tf/doc_len
            # 2. calculate p(w|D)
            try:
                tf = doc.get_term_freq_dict()[term]
                prob_w_D = tf / doc_len                    # prob_w_D is the probability of the word w in document D
            except KeyError:
                prob_w_D = 0
            # 3. calculate p(w|D) * p(Q|D)
            prob_w_r += prob_w_D * prob_Q_D

        term_likelihood[term] = prob_w_r

    # sort dictionary in descending order of term relevance model prob
    term_likelihood_sort = {key: value for key, value in
                            sorted(term_likelihood.items(), key=lambda item: item[1], reverse=True)}
    
    return term_likelihood_sort

def kl_divergence_score(docColl, term_likelihood_prob):

    """
    Calculate the KL-divergence score for each doc in the docColl

    Params
    -------
    term_likelihood_prob    : output from relevance_model_prob()
    docColl                 : document collection

    Returns
    -------
    {docid: kl_score, ...} sorted dictionary containing kl-score for each document 
    """

    kl_rank = {}
    for docid, doc in docColl.get_docs().items(): # for each doc in the docColl
        kl_score = 0
        for term in doc.get_term_list():
            #P(w|D)
            try:
                prob_w_D = doc.get_term_freq_dict()[term] / doc.get_doc_len()
                kl_score += term_likelihood_prob[term] * math.log(prob_w_D)
            except KeyError:
                kl_score = 0
        if kl_score != 0:
            kl_rank[docid] = - kl_score
        else:
            kl_rank[docid] = kl_score  # when kl_score is 0
    
    # sort the KL rank by the kl_score in ascending order 
    kl_rank_sort = {key: value for key, value in sorted(kl_rank.items(), key=lambda item: item[1], reverse=True)}
    
    return kl_rank_sort


####################################################### Main Function #############################################################

if __name__ == '__main__':

    # parse documents to get Bag_of_words representation
    curr_dir = os.getcwd()
    query_f = 'Queries.txt'
    stopwords_f = open('common-english-words.txt', 'r')
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()

    # parse queries and instantiate query collection
    queryColl = util.parse_query(query_f, stop_words)
    os.chdir(curr_dir)

    # create output directories
    os.makedirs('Outputs/Baseline/Rankings/')
    os.makedirs('Outputs/Baseline/Relevance/')
    os.makedirs('Outputs/My_model1/Rankings/')
    os.makedirs('Outputs/My_model1/Relevance/')
    os.makedirs('Outputs/My_model2/Rankings/')
    os.makedirs('Outputs/My_model2/Relevance/')

    os.chdir('DataSets/')

    for dataset_name in glob.glob('*'):

        # get dataset id 
        dataset_id = 'R' + re.search(r'\d+', dataset_name).group()
        # print(dataset_id)
        # create document collection of BowDoc objects
        docColl = util.parse_rcv_coll(dataset_name, stop_words)
        os.chdir('..')
        # set doc frequency, cummulative term frequency, average and total document length
        docColl.set_doc_freq(util.calc_df(docColl)) 
        docColl.set_cum_freq(util.calc_cum_tf(docColl)) 
        docColl.set_avg_doc_len(util.avg_doc_len(docColl)) 
        docColl.set_total_doc_len(util.total_doc_len(docColl)) 
        # get the query object from the queryColl
        query = queryColl.get_query(dataset_id)

        #################################################################################################
        # MODEL 1 - Baseline BM25
        #################################################################################################
        # get sorted document scores
        bm25_score = bm25(docColl, query.get_term_freq_dict())

        # write document rankings and relevance labels
        output_file_score = open('../Outputs/Baseline/Rankings/Baseline_' + dataset_id + 'Ranking.dat', 'w')
        output_file_score.write(f"Query{dataset_id[1:]} (DocID Weight):\n")
        output_file_rel = open('../Outputs/Baseline/Relevance/Dataset' + dataset_id[1:] + '.txt', 'w') 

        count = 1
        for docid, score in bm25_score.items():
            if count < 13:
                output_file_rel.write(f"{dataset_id} {docid} 1\n")
                output_file_score.write(f"{docid} {score}\n")
            else:
                output_file_rel.write(f"{dataset_id} {docid} 0\n")
            count += 1
        output_file_score.close()
        output_file_rel.close()

        ##################################################################################################
        # MODEL 2 - Rocchio Query Expansion 
        ##################################################################################################

        # Step 1: Calculate document relevance score using TF-IDF IR model
        tfidf_score = tfidf_IR(query.get_term_freq_dict(), docColl)

        # Step 2: Create pseudo-relevance feedback and term vocabulary based on relevant document set
        tfidf_pos = []
        tfidf_neg = []
        tfidf_vocab = query.get_term_list()

        k_counter = 1
        for docid, score in tfidf_score.items():

            # take top 10 as the relevant documents
            if k_counter < 11:
                tfidf_pos.append(docid)
                # create union set of terms in all relevant docs
                for term in docColl.get_doc(docid).get_term_list():
                    if term not in tfidf_vocab:
                        tfidf_vocab.append(term)
            # otherwise non-relevant
            else:
                tfidf_neg.append(docid)
            k_counter += 1
        
        # Step 3: Run rocchio query expansion and get the top 50 terms with the highest average weights as the expanded query
        top50_queryTerms = rocchio(query.get_term_freq_dict(), tfidf_vocab, tfidf_pos, tfidf_neg, docColl)

        # Step 4: Run the tfidf_IR() with the new query terms and weights
        rocchio_score = tfidf_IR(top50_queryTerms, docColl)

        # # write document rankings and relevance labels
        output_file_score = open('../Outputs/My_model1/Rankings/Model1_' + dataset_id + 'Ranking.dat', 'w')
        output_file_score.write(f"Query{dataset_id[1:]} (DocID Weight):\n")
        output_file_rel = open('../Outputs/My_model1/Relevance/Dataset' + dataset_id[1:] + '.txt', 'w') 

        count = 1
        for docid, score in rocchio_score.items():
            if count < 13:
                output_file_rel.write(f"{dataset_id} {docid} 1\n")
                output_file_score.write(f"{docid} {score}\n")
            else:
                output_file_rel.write(f"{dataset_id} {docid} 0\n")
            count += 1
        output_file_score.close()
        output_file_rel.close()

        # #################################################################################################
        # # MODEL 3 - Query-Likelihood Model + KL Divergence Score
        # #################################################################################################

        # step 1: rank documents by query likelihood score for query 
        ql_sorted = query_likelihood(query.get_term_list(), docColl)

        # step 2: select top-k ranked documents to be relevant set and calculate the 
        # relevance model probabilities P(w|R) for each term in the collection
        relevance_prob = relevance_model_prob(ql_sorted, query.get_term_list(), docColl, 10)

        # step 3: rank documents again using KL-divergence score 
        kl_score = kl_divergence_score(docColl, relevance_prob)

        # write document rankings and relevance labels
        output_file_score = open('../Outputs/My_model2/Rankings/Model2_' + dataset_id + 'Ranking.dat', 'w')
        output_file_score.write(f"Query{dataset_id[1:]} (DocID Weight):\n")
        output_file_rel = open('../Outputs/My_model2/Relevance/Dataset' + dataset_id[1:] + '.txt', 'w') 

        count = 1
        for docid, score in kl_score.items():
            if count < 13:
                output_file_rel.write(f"{dataset_id} {docid} 1\n")
                output_file_score.write(f"{docid} {score}\n")
            else:
                output_file_rel.write(f"{dataset_id} {docid} 0\n")
            count += 1
        output_file_score.close()
        output_file_rel.close()




