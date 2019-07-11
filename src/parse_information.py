# -*- coding: utf-8 -*-

"""
@author: jsuter
Project: STIMPY - Sentence Transformative Infernece Mapping for Python

Julia Suter, 2019
---
parse_information.py

Classes for Sentence and Token for easy processing of CoNNL formatted output
given by ParZu and CorZu.

- Token class, 
- Sentence class,
- get_dependent_tokens

"""

# ----------------------------------------
### Import Statements
# ----------------------------------------

from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# --- Classes for Sentence and Token --- 

class Sentence(object):    
    """Sentence Class for sentences parsed by ParZu, or in CoNNL format
    Offers many methods for returning syntatical functions of Sentence like subject, predicate or prepositional phrase
    """
    
    def __init__(self,data):
            self.data = data
    

class Token(object):
    
    """Token Class for representing Tokens in sentences parsed by ParZu or in CoNNL format.
    Offers many functions for returning linguistic information on token
    and changing it.
    """
    def __init__(self,token_data):    

            self.data = token_data
            
            self.id = int(token_data[0])
            self.form = token_data[1]
            self.lemma = token_data[2]
            self.u_pos = token_data[3]
            self.x_pos = token_data[4]
            self.head = int(token_data[5])
            self.deprel = token_data[6]
            self.polarity = 'up'
            
            # specific scope
            self.specific_projectivity = None
            
            
    
    def same_token(self, other): 
        """"Check whether it is the same token by form, lemma, pos"""
        return self.form == other.form and self.lemma == other.lemma \
                        and self.u_pos == other.u_pos \
                        and self.x_pos == other.x_pos \
                        and self.deprel == other.deprel # and self.head == other.head
                    
        
def get_dependent_tokens(sent,head,already_processed = []):
    """Return all words in sentence that are dependent on head.
    
    Args:       sent (list), head (Token)
    Returns:    dependent_tokens (list of Tokens)
    """

    # Get all dependent tokens
    dependent_tokens = [k for k in sent if (k.head == head.id) \
                                        and k not in already_processed] 

    # If no dependent tokens, return empty list
    if not dependent_tokens:
        return []
        
    # For all dependent tokens, get their dependent tokens (recursion)
    else:
        for k in dependent_tokens:             
                new_dep_tokens = get_dependent_tokens(sent,k, dependent_tokens)
                dependent_tokens += new_dep_tokens
 
        return dependent_tokens
  

