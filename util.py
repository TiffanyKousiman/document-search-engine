import glob, os, re
import string
from stemming.porter2 import stem

class BowDoc:
    """Bag-of-words representation of a document.

    The document has an ID, and an iterable list of terms with their
    frequencies."""

    def __init__(self, docid):
        """Constructor.

        Set the ID of the document, and initiate an empty term dictionary.
        Call add_term to add terms to the dictionary."""
        self.docid = docid
        self.terms = {}
        self.doc_len = 0

    def add_term(self, term):
        """Add a term occurrence to the BOW representation.

        This should be called each time the term occurs in the document."""
        try:
            self.terms[term] += 1
        except KeyError:  
            self.terms[term] = 1

    def get_term_count(self, term):
        """Get the term occurrence count for a term.

        Returns 0 if the term does not appear in the document."""
        try:
            return self.terms[term]
        except KeyError:
            return 0

    def get_term_freq_dict(self):
        """Return dictionary of term:freq pairs."""
        return self.terms

    def get_term_list(self):
        """Get sorted list of all terms occurring in the document."""
        return sorted(self.terms.keys())

    def get_docid(self):
        """Get the ID of the document."""
        return self.docid

    def __iter__(self):
        """Return an ordered iterator over term--frequency pairs.

        Each element is a (term, frequency) tuple.  They are iterated
        in term's frequency descending order."""
        return iter(sorted(self.terms.items(), key=lambda x: x[1],reverse=True))
        """Or in term alphabetical order:
        return iter(sorted(self.terms.iteritems()))"""
        
    def get_doc_len(self):
        return self.doc_len

    def set_doc_len(self, doc_len):
        self.doc_len = doc_len

class BowColl:
    """Collection of BOW documents."""

    def __init__(self):
        """Constructor.

        Creates an empty collection."""
        self.docs = {}
        self.doc_freq = {} # term : doc frequency pairs 
        self.cum_tf = {}
        self.avg_doc_len = 0
        self.total_doc_len = 0

    def add_doc(self, doc):
        """Add a document to the collection."""
        self.docs[doc.get_docid()] = doc

    def get_doc(self, docid):
        """Return a document by docid.

        Will raise a KeyError if there is no document with that ID."""
        return self.docs[docid]

    def get_docs(self):
        """Get the full list of documents.

        Returns a dictionary, with docids as keys, and docs as values."""
        return self.docs

    def inorder_iter(self):
        """Return an ordered iterator over the documents.
        
        The iterator will traverse the collection in docid order.  Modifying
        the collection while iterating over it leads to undefined results.
        Each element is a document; to find the id, call doc.get_docid()."""
        return BowCollInorderIterator(self)

    def get_num_docs(self):
        """Get the number of documents in the collection."""
        return len(self.docs)

    def __iter__(self):
        """Iterator interface.

        See inorder_iter."""
        return self.inorder_iter()
    
    def set_doc_freq(self, doc_freq):
        """Set dictionary of term:df pairs for all terms in the collection"""
        self.doc_freq = doc_freq
    
    def get_doc_freq(self):
        """Return term : document frequency dictionary"""
        return self.doc_freq
    
    def set_cum_freq(self, cum_tf):
        """Set dictionary of term:cum_tf pairs for all terms in the collection"""
        self.cum_tf = cum_tf
    
    def get_cum_freq(self):
        """Return term : cummulative term frequency dictionary"""
        return self.cum_tf
    
    def set_avg_doc_len(self, avg_len):
        """Set average document length of all documents in the collection"""
        self.avg_doc_len = avg_len
    
    def get_avg_doc_len(self):
        """Return average document length"""
        return self.avg_doc_len
    
    def set_total_doc_len(self, total_len):
        """Set total document length of all documents in the collection"""
        self.total_doc_len = total_len
    
    def get_total_doc_len(self):
        """Return total document length"""
        return self.total_doc_len

class BowQuery:
    """Bag-of-words representation of a query.

    Each query has a number and an iterable list of query terms with their
    frequencies."""

    def __init__(self, q_id):
        """Constructor.

        Set the ID of the query, and initiate an empty term dictionary.
        Call add_term to add terms to the dictionary."""
        self.id = q_id
        self.terms = {}

    def add_term(self, term):
        """Add a term occurrence to the BOW representation.

        This should be called each time the term occurs in the query."""
        try:
            self.terms[term] += 1
        except KeyError:  
            self.terms[term] = 1

    def get_term_count(self, term):
        """Get the term occurrence count for a term.

        Returns 0 if the term does not appear in the query."""
        try:
            return self.terms[term]
        except KeyError:
            return 0

    def get_term_freq_dict(self):
        """Return dictionary of term:freq pairs."""
        return self.terms

    def get_term_list(self):
        """Get sorted list of all terms occurring in the query."""
        return sorted(self.terms.keys())

    def get_id(self):
        """Get the ID of the query."""
        return self.id

    def __iter__(self):
        """Return an ordered iterator over term--frequency pairs.

        Each element is a (term, frequency) tuple.  They are iterated
        in term's frequency descending order."""
        return iter(sorted(self.terms.items(), key=lambda x: x[1],reverse=True))
        """Or in term alphabetical order:
        return iter(sorted(self.terms.iteritems()))"""

class QueryColl:
    """Collection of BOW queries."""

    def __init__(self):
        """Constructor.

        Creates an empty collection."""
        self.queries = {}

    def add_query(self, query):
        """Add a query to the collection."""
        self.queries[query.get_id()] = query

    def get_query(self, q_id):
        """Return a query by id"""
        return self.queries[q_id]

    def get_queries(self):
        """Get the full list of queries.
        Returns a dictionary, with id as keys, and query as values."""
        return self.queries

class BowCollInorderIterator:
    """Iterator over a collection."""

    def __init__(self, coll):
        """Constructor.
        
        Takes the collection we're going to iterator over as sole argument."""
        self.coll = coll
        self.keys = sorted(coll.get_docs().keys())
        self.i = 0

    def __iter__(self):
        """Iterator interface."""
        return self

    def next(self):
        """Get next element."""
        if self.i >= len(self.keys):
            raise StopIteration
        doc = self.coll.get_doc(self.keys[self.i])
        self.i += 1
        return doc

def parse_rcv_coll(inputpath, stop_words):
    """Parse an RCV1 data files into a collection.

    inputpath is the folder name of the RCV1 data files.  The parsed collection
    is returned.  NOTE the function performs very limited error checking."""
    #stopwords = open('common-english-words.txt', 'r')
    
    coll = BowColl()    
    os.chdir(inputpath)
    for file_ in glob.glob("*.xml"):
        curr_doc = None
        start_end = False
        word_count = 0
        for line in open(file_):
            line = line.strip()
            if(start_end == False):
                if line.startswith("<newsitem "):
                    for part in line.split():
                        if part.startswith("itemid="):
                            docid = part.split("=")[1].split("\"")[1]
                            curr_doc = BowDoc(docid)
                            break
                    continue    
                if line.startswith("<text>"):
                    start_end = True  
            elif line.startswith("</text>"):
                break
            elif curr_doc is not None:
                line = line.replace("<p>", "").replace("</p>", "")
                line = re.sub("\\s+", " ", line) 
                line = line.translate(str.maketrans('','', string.digits)).\
                    translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))

                for term in line.split():
                    word_count += 1
                    term = stem(term.lower())
                    if len(term) > 2 and term not in stop_words:
                        curr_doc.add_term(term)
        if curr_doc is not None:
            curr_doc.set_doc_len(word_count)
            coll.add_doc(curr_doc)

    return coll

def parse_query(query_file, stop_words):

    """
    Parse all queries in a file into a query collection.
    query term is only indexed if it is at least 3 or more chars long, and not a stop word. 

    Params
    -------
    query_file                  : a file of queries
    stop_words (list of str)    : a list of stop words 
    """

    query_coll = QueryColl()
    curr_query = None

    for line in open(query_file):

        # get query_id and use it to instantiate query object 
        if line.startswith("<num>"):
            id_ = line.split("<num>")[1].split(':')[1].rstrip().strip()
            curr_query = BowQuery(id_)
        # get query terms and frequency from the title tag only
        if line.startswith("<title>"):                              # Split after the title
            title = line.split("<title>")[1].rstrip().strip()       # Take away any trailing newline characters and also any excessive white space
            title = re.sub("\\s+", " ", title)                      # standardize whitespaces
            title = title.translate(str.maketrans('','', string.digits)).\
                translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))  # discard punctuations and number
            for term in title.split():
                term = stem(term.lower()) # stemming and convert all chars to lower-case
                # add words with qualified qualities defined in the docstring (at least 3 characters and not a stop word) to `docWord`
                if len(term) > 2 and term not in stop_words:
                    curr_query.add_term(term)
    
        if curr_query is not None:
            query_coll.add_query(curr_query)

    return query_coll

def calc_df(coll):
    """
    calculate document-frequency (df) for a given BowColl collection
    and return a {term:df, ...} dictionary
    """
    df_ = {}

    for _, doc in coll.get_docs().items():
        for term in doc.get_term_list():
            try: 
                df_[term] += 1
            except KeyError:
                df_[term] = 1
    
    return df_

def calc_cum_tf(coll):
    
    """calculate the cumulative term frequency for all terms in the BowColl collection
    and return a {term: cum_freq, ...} dictionary """

    cum_tf = {}

    for _, doc in coll.get_docs().items():
        for term, tf in doc.get_term_freq_dict().items():
            try:
                cum_tf[term] += tf
            except KeyError:
                cum_tf[term] = tf
    
    return cum_tf

def total_doc_len(coll):
    """
    calculate total document lengths of all documents in the collection
    """
    total_dl = 0
    for _, doc in coll.get_docs().items():
        total_dl = total_dl + doc.get_doc_len()
        
    return total_dl

def avg_doc_len(coll):
    """
    calculate the average document lengths of all documents in the collection.
    """
    return total_doc_len(coll)/coll.get_num_docs()