import math
import pandas as pd
import os 

def get_re_docs(truth_path, eval_path):

    """
    get relevant documents A and retrieved documents B
    
    Params
    --------
    truth_path: file path for relevance benchmark (truth)
    eval_path:  file path for evaluated relevance (model outputs)

    Returns
    --------
    Tuple of (A,B). 
        A is the dictionary of {docid:label, ...} of both relevant and non-relevant docs, 
        B is the dictionary of {docid:rank,...} of all retrieved docs (labelled as 1)
    """

    A = {}
    B = {}

    for line in open(truth_path):
        line = line.strip()
        line1 = line.split()
        A[line1[1]] = int(float(line1[2]))
    
    r = 1 # ranking of retrieved doc
    for line in open(eval_path):
        line = line.strip()
        line1 = line.split()
        if int(float(line1[2])) == 0:
            break
        else:
            B[line1[1]] = r
            r += 1

    return (A,B)

def calc_avg_precision(rel_docs, retrieved_docs):

    ri = 0      # number of retrieved docs that are relevant
    total_precision = 0.0

    # find the number of relevant documents and retrieved docs
    total_rel = len([id for (id,v) in rel_docs.items() if v > 0])
    total_retrieved = len(retrieved_docs)

    # calculate the map 
    for docid, n in retrieved_docs.items():
        if rel_docs[docid] > 0:                 # if retrieved doc is relevant 
            ri += 1                             # number of retrieved relevant doc at rank i
            pi = float(ri)/float(n)
            total_precision += pi
            # print(f"At position {n}, precision = {pi}")
    
    if total_precision > 0:
        avg_precision = round(total_precision/float(ri), 2)
    else:
        avg_precision = total_precision

    # print(f"The average precision: {avg_precision}")

    return avg_precision


def calc_precision_at_12(rel_docs, retrieved_docs):
    """Calculate the precision at rank 12"""
    # number of documents in the retrieved_docs that are actually relevant
    r = 0
    for doc, n in retrieved_docs.items():
        if rel_docs[doc] > 0:
            r += 1
        if n == 12: # break the iterator at rank 12 
            break

    return round(r/12, 2)

def calc_DCG_at_12(rel_docs, retrieved_docs):
    
    """Calculate the discounted cummulative gain at rank position 12"""
    
    cum_gain = 0
    for doc, n in retrieved_docs.items():
        # get the true relevance score - 1 or 0 
        if rel_docs[doc] > 0:
            rel = 1
        else:
            rel = 0
        # calculate the discounted and cummative gain
        if n == 1:
            cum_gain += rel
        else:
            cum_gain += rel/math.log2(n)
        # end the for loop at position 12
        if n == 12:
            break

    # print("Discounted cummative gain at rank position 12: ", round(cum_gain, 2))

    return round(cum_gain,2)

def evaluate(model_name, dataset_id, truth_path, average_precision, precision_at_12, dcg_at_12):

    topic_id = 'R' + dataset_id

    eval_path = f'Outputs/{model_name}/Relevance/Dataset{dataset_id}.txt'
    (rel_docs, retrieved_docs) = get_re_docs(truth_path, eval_path)
    avg_pre = calc_avg_precision(rel_docs, retrieved_docs)
    pre_12 = calc_precision_at_12(rel_docs, retrieved_docs)
    dcg_12 = calc_DCG_at_12(rel_docs, retrieved_docs)
    average_precision[model_name][topic_id] = avg_pre
    precision_at_12[model_name][topic_id] = pre_12
    dcg_at_12[model_name][topic_id] = dcg_12

####################################################### Main Function #############################################################

if __name__ == '__main__':

    average_precision = {}
    precision_at_12 = {}
    dcg_at_12 = {}

    # model evaluation
    for model in ['Baseline', 'My_model1', 'My_model2']:
        average_precision[model] = {}
        precision_at_12[model] = {}
        dcg_at_12[model] = {}

        for n in range(101, 151):
        
            dataset_id = str(n)
            truth_path = f'Feedback/Dataset{dataset_id}.txt'
            evaluate(model, dataset_id, truth_path, average_precision, precision_at_12, dcg_at_12)
    
    # export results to excel files
    os.makedirs('Outputs/Excel/')

    avg_precision_df = pd.DataFrame(average_precision)
    avg_precision_df.loc['Mean Average Precision'] = avg_precision_df.mean()
    avg_precision_df.to_excel('Outputs/Excel/AveragePrecision.xlsx', index=True)

    precision_12_df = pd.DataFrame(precision_at_12)
    precision_12_df.loc['Average Precision'] = precision_12_df.mean()
    precision_12_df.to_excel('Outputs/Excel/PrecisionAt12.xlsx', index=True)

    dcg_12_df = pd.DataFrame(dcg_at_12)
    dcg_12_df.loc['Average DCG at 12'] = dcg_12_df.mean()
    dcg_12_df.to_excel('Outputs/Excel/DCGAt12.xlsx', index=True)