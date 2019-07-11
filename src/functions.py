# -*- coding: utf-8 -*-
"""
@author: jsuter
Project: STIMPY - Sentence Transformative Infernece Mapping for Python

Julia Suter, 2019
---
functions.py

- Operator and pre-fixed WordNet dictionaries
- Help functions to retrieve related words and inflections
- Predicate functions
- Transition functions: delete, select, insert, replace
"""

# ----------------------------------------
### IMPORTS
# ----------------------------------------

# Import Statements
from pattern3.en import conjugate as conj
from pattern3.en import pluralize as plur
from pattern3.en import comparative, superlative

import re
import pickle

# Own modules
import parse_information as easy_parse
import RUN_stimpy as stimpy
import relations_and_polarity_settings as rel_pol
import rule_reading_system as rrs


# ----------------------------------------
### SETTINGS: WordNet relations for toy examples, operators
# ----------------------------------------
        
# WordNet dicts
hypernyms = {}
hyponyms = {}
siblings = {}
synonyms = {}


## Wordnet knowledge for Toy Examples

hypernyms = {('poodle','NOUN'):['dog'],
             ('dog','NOUN'):['animal'],
            ('cat','NOUN') : ['animal','carnivore'], 
             ('mouse','NOUN') :['animal'],
             ('salmon','NOUN') :['fish'],
             ('swede','NOUN') :['scandinavian']}
                         
hyponyms =  {('dog','NOUN') :['poodle'],
             ('carnivore','NOUN') :['cat'],
             ('animal','NOUN') : ['mouse','cat','dog'],
             ('fish','NOUN') :['salmon'],
             ('scandinavian','NOUN') :['swede']}
             
siblings = {('dog','NOUN') :['cat'],
            ('cat','NOUN') :['dog']}
         
synonyms = {('cat','NOUN') :['kitty'],
            ('eat','VERB') :['devour'],
            ('devour','VERB') :['eat'],
            ('big','ADJ'):['great']}

antonyms = {('hate','VERB'):['love'],
            ('love','VERB'):['hate']}

# Operators

operators = ['some','all','every','no']

plural_operators = ['all','most','many','some','several','few','no','both']

both_operators = {'both':2,
                  'either':1,
                  'neither':0}

# Coverage dict for operaors
general_operators = {'all':10,
                     'every':10,
                     'each':10,
                     'both':10,
                     
                     'most':9,
                     'majority':9,
                     
                     'many':8,
                     'a lot':8,
                     'plenty':8,
                     'large number':8,
                     'great number':8,
                     'enough':8,
                     
                     'some':5,
                     'a couple':5,
                     'several':5,
                     'a number of':5,
                     
                     'few':2,
                     'a little':2,
                     'a bit':2,
                     'no':0,
                     'none':0,
                     'neither':0,                     
                }


# ----------------------------------------
### HELP FUNCTIONS TO GET RELATIONS, INFLECTIONS, and NEW PARSE
# ----------------------------------------

def load_wordnet_relations():
    """Load WordNet relations from dict."""

    # Load synonyms
    with open('./wordnet_dicts/synonyms_dict.pkl', 'rb') as fp:
        global synonyms
        synonyms = pickle.load(fp)
        
    # Load hypernyms
    with open('./wordnet_dicts/hypernyms_dict.pkl', 'rb') as fp:
        global hypernyms
        hypernyms = pickle.load(fp)
        
    # Load hyponyms
    with open('./wordnet_dicts/hyponyms_dict.pkl', 'rb') as fp:
        global hyponyms
        hyponyms = pickle.load(fp)
    
    # Load antonyms
    with open('./wordnet_dicts/antonyms_dict.pkl', 'rb') as fp:
        global antonyms
        antonyms = pickle.load(fp)


def reparse(premise):
    """Reparse premise."""
    
    # Get tokens
    new_premise_tokens = premise.tokens
    # Parse sentence
    premise_parse = stimpy.nlp(" ".join([t.form for t in new_premise_tokens]))
    # Get tokens and sent
    premise_tokens, premise_sent = stimpy.get_tokens_and_sent(premise_parse)
    # Get Premise instance
    premise = stimpy.Premise(premise_tokens, premise_sent, state=premise.state, validity=premise.validity, other_premises=premise.other_premises)
    
    return premise


def get_inflected_word(original_word, pos):
    """Return inflected word according to given pos"""
    
    # Get word to be inflected
    if isinstance(original_word,str):
        word = original_word
    else:
        word = original_word.lemma

        # Do not alter some pos
        if original_word.x_pos in ['WP','WDT']:
            return word
        
        # Compute correct form for "be"
        if word == 'be' and pos == 'VBD' and original_word.form == 'was' and original_word.x_pos == 'VBD':
            return 'were'
        
        if word == 'be' and pos == 'VBD' and original_word.form == 'were' and original_word.x_pos == 'VBD':
            return 'was'
        
    
    # Unchangable word classes
    if pos in ['CD','DT','VB','NN','NNP','JJ','RB','RP','IN','CC','PRP',
               'PRP$', 'AFX','MD','WM','WP','WP$','TO','$',"''", "``",',',':',
               'UH','WRB','FW','WDT','RBR','PDT','XX','POS','RBS','EX','ADD',
               'NFP','LS','SYM','HYPH','.','-RRB-','-LRB-']: 
        return word
                
    # ~~~ change nouns, adjectives and verbs ~~~  

    if pos in ['NNS','NNPS']:
        new_word = plur(word)
        
    elif pos == 'VB':
        new_word = conj(word, "inf")
               
    elif pos == 'VBP':
        new_word = conj(word, "2sg")
        
    elif pos == 'VBZ':
        new_word = conj(word, "3sg")
        
    elif pos == 'VBG':
        new_word = conj(word, "part")
        
    elif pos == 'VBN':
        new_word = conj(word, "ppart")
        
    elif pos == 'VBD':
        new_word = conj(word, "p")
            
    elif pos == 'JJR':
        new_word = comparative(word)       
        
    elif pos == 'JJS':
        new_word = superlative(word)
            
    # Return inflected word
    return new_word
 

# ----------------------------------------
### PREDICATE FUNCTIONS
# ----------------------------------------

def operator_replacement_action(args, premise, verbose=True):
    """Compare two operators and return function for adjusting first
    argument number to second argument number: pluralize, singularize, none"""
    
    # Get operators
    op1, op2 = args
    
    # Are they plural operators?
    op1_plural_operator = op1.lemma in plural_operators
    op2_plural_operator = op2.lemma in plural_operators

    
    # Return corresponding function
    if op1_plural_operator == op2_plural_operator:
        return 'none'
    
    if op1_plural_operator and not op2_plural_operator:
        return 'singularize'
    
    if not op1_plural_operator and op2_plural_operator:
        return 'pluralize'
        

def pluralize(args, premise):
    """Return plural equivalent POS-tag for given POS-tag."""

    pos = args[0]
    
    # Pluralize dict
    pluralize_dict = {'NN':'NNS',
                      'NNS':'NNS',
                      'VBZ':'VBP',
                      'VBD':'VBD',
                      'VB':'VB',
                      'VBP':'VBP',
                      'NNP':'NNPS',
                      'VBN':'VBN',
                      'MD':'MD',}
    
    # Test whether POS exists in dict, otherwise return unaltered pos
    try:
        plural_pos = pluralize_dict[pos]
    except KeyError:
        return pos
    return plural_pos


def singularize(args, premise):
    """Return singular equivalent POS-tag for given POS-tag."""  

    pos = args[0]
    
    # Singularize dict
    singularize_dict = {'NNS':'NN',
                      'NN':'NN',
                      'NNPS':'NNP',
                      'VBP':'VBZ',
                      'VBD':'VBD',
                      'VB':'VB',
                      'VBZ':'VBZ',
                      'VBN':'VBN',
                      'MD':'MD'   }
    
    # Test whether pos is in dict, otherwise return unaltered pos
    try:
        sing_pos = singularize_dict[pos]
    except KeyError:
        return pos
    return sing_pos

               
        
def positive_sent(args, premise, verbose=True):
    """Check whether root verb is negated"""
    
    # If 'not' is in the lemma list
    if 'not' in premise.lemmas:     
        
        # Get all "not"
        NOT_tokens = [t for t in premise.tokens if t.lemma == 'not']
        
        for not_t in NOT_tokens:
            # Get root verb
            root_verb = next(t for t in premise.tokens if t.deprel == 'ROOT')
            
            # Is "not" dependent on root verb?
            if not_t.head == root_verb.id:
                return False
            
        return True   
    
    # If no "not", assume it is positive sentence
    else:
        return True
    
def operator_comparison(args, premise, verbose=True):
    """Compare operators and check whether first operator is stronger,
    weaker or equal."""
    
    # Get operators
    op_2, op_1 = args

    # Get lemmas
    op_1 = op_1.lemma
    op_2 = op_2.lemma

    # Get operator strengths
    if op_1 in both_operators and op_2 in both_operators:
            op_1_strength = both_operators[op_1]
            op_2_strength = both_operators[op_2]

    # Get operator strengths          
    elif op_1 in general_operators and op_2 in general_operators:        
            op_1_strength = general_operators[op_1]
            op_2_strength = general_operators[op_2]
    
    # Unknown operators error      
    else:
        raise IOError('One or more unknown operators: ' + op_1 + ', ' +op_2)

    # Return comparison value  
    if op_1_strength > op_2_strength:
        return 'stronger'    
    elif op_1_strength == op_2_strength:
        return 'equal'    
    else:
        return 'weaker'
            

def exists_hypernym(args, premise, verbose=True):
    """Check whether there is a matching hypernym."""
    
    # Get token
    token = args[0]
       
    # Check whether there exists a hpernym
    if (token.lemma, token.u_pos) in hypernyms.keys():
        premise.TRANS.available_hypernyms = len(hypernyms[(token.lemma, 
                                                             token.u_pos)])
        return True
    else:
        return False        
        
def exists_hyponym(args, premise, verbose=True):
    """Check whether there is a matching hyponym."""
    
    # Get token
    token = args[0]
   
    # Check whether there exists a hyponym
    if (token.lemma, token.u_pos) in hyponyms.keys():
        premise.TRANS.available_hyponyms = len(hyponyms[(token.lemma, 
                                                           token.u_pos)])
        return True
    else:
        return False    
        
def exists_sibling(args, premise, verbose=True):
    """Check whether there is a matching sibling."""
    
    # Get token
    token = args[0]
   
    # Check whether there exists a sibling
    if (token.lemma, token.u_pos) in siblings.keys():
        premise.TRANS.available_siblings = len(siblings[(token.lemma,
                                                           token.u_pos)])
        return True
    else:
        return False        
        
def exists_synonym(args, premise, verbose=True):
    """Check whether there is a matching synonym."""
    
    # Get token
    token = args[0]
   
    # Check whether there exists a synonym
    if (token.lemma, token.u_pos) in synonyms.keys():
        premise.TRANS.available_synonyms = len(synonyms[(token.lemma, 
                                                           token.u_pos)])
        return True
    else:
        return False        
        
  
def exists_antonym(args, premise, verbose=True):
    """Check whether there is a matching synonym."""
    
    # Get token
    token = args[0]
   
    # Check whether there exists an antonym
    if (token.lemma, token.u_pos) in antonyms.keys():
        premise.TRANS.available_antonyms = len(antonyms[(token.lemma, 
                                                           token.u_pos)])
        return True
    else:
        return False       

        
def get_synonym(args, premise, verbose=True):
    """Get synonym for given token."""

    # Get token
    token = args[0]
    
    # Get synonym according to lemma, pos and number
    synonym = synonyms[(token.lemma, token.u_pos)][premise.TRANS.synonym_nr]
    
    # Check whether this word/synonym pair was already processed
    if (premise.words, token.lemma, token.u_pos, synonym) in premise.TRANS.processed_synonym_pairs:
        return None
    
    # If not: save, clean and return synonym
    else:
        premise.TRANS.processed_synonym_pairs.append((premise.words, 
                                                        token.lemma, 
                                                        token.u_pos, synonym))
        synonym = re.sub('_',' ',synonym)
        return synonym 
            
    return None
        
def get_hyponym(args, premise, verbose=True):
    """Get hyponym for given token."""

    # Ge token
    token = args[0]
    
    # Get hypnym according to lemma, pos and number
    hyponym = hyponyms[(token.lemma, token.u_pos)][premise.TRANS.hyponym_nr]
    
    # Check whether this word/hyponym pair was already processed
    if (premise.words, token.lemma, token.u_pos, hyponym) in premise.TRANS.processed_hyponym_pairs:
        return None
    
    # If not: save, clean and return hyponym
    else:
        premise.TRANS.processed_hyponym_pairs.append((premise.words, 
                                                        token.lemma, 
                                                        token.u_pos, hyponym))
        hyponym = re.sub('_',' ',hyponym)
        return hyponym
        
def get_hypernym(args, premise, verbose=True):
    """Get hypernym for given token."""
    
    # Get token
    token = args[0]
    
    # Get hypernym according to lemma, pos and number
    hypernym = hypernyms[(token.lemma, token.u_pos)][premise.TRANS.hypernym_nr]
    
    # Check whether this word/hypernym pair was already processed
    if (premise.words, token.lemma, token.u_pos, hypernym) in premise.TRANS.processed_hypernym_pairs:
        return None
    
    # If not: save, clean and return hypernym
    else:
        premise.TRANS.processed_hypernym_pairs.append((premise.words, 
                                                         token.lemma, 
                                                         token.u_pos, hypernym))
        hypernym = re.sub('_',' ',hypernym)
        return hypernym
        
def get_sibling(args, premise, verbose=True):
    """Get sibling for given token."""
    
    # Get token    
    token = args[0]
    
    # Get sibling according to lemma, pos and number
    sibling = siblings[(token.lemma, token.u_pos)][premise.TRANS.sibling_nr]
    
    # Check whether this word/sibling pair was already processed
    if (premise.words, token.lemma, token.u_pos, sibling) in premise.TRANS.processed_sibling_pairs:
        return None
    
    # If not: save, clean and return sibling
    else:
        premise.TRANS.processed_sibling_pairs.append((premise.words,
                                                        token.lemma, 
                                                        token.u_pos, sibling))
        sibling = re.sub('_',' ',sibling)
        return sibling
    
def get_antonym(args, premise, verbose=True):
    """Get sibling for given token."""
    
    # Get token
    token = args[0]
    
    # Get antonym according to lemma, pos and number
    antonym = antonyms[(token.lemma, token.u_pos)][premise.TRANS.antonym_nr]
    
    # Check whether this word/antonym pair was already processed
    if (premise.words, token.lemma, token.u_pos, antonym) in premise.TRANS.processed_antonym_pairs:
        return None
    
    # If not: save, clean and return antonym
    else:
        premise.TRANS.processed_antonym_pairs.append((premise.words, 
                                                        token.lemma, 
                                                        token.u_pos, antonym))
        antonym = re.sub('_',' ',antonym)
        return antonym

def is_passive(args, premise, verbose=True):
    """Get whether the sentence or phrase contains a passive construction."""
    
    is_passive = False
    
    # Get dependent tokens
    deprels = [t.deprel for t in premise.tokens]
    
    # Check whether specific dependency relations are contained in dependent tokens
    if 'auxpass' in deprels and 'nsubjpass' in deprels:
        is_passive = True
        
    return is_passive

def same_phrase_negated(args, premise, verbose=True):
    """Not used anymore."""
    return None

def same_phrase(args, premise, verbose=True):
    """Check whether two phrases are the equal, or equal but negated."""
  
    negated = False
    
    # Get other premise
    other_premise = premise.other_premise   
    
    ## Get anchor tokens, and is_negated value
    if len(args)==2:
        current_prem_anchor_token, other_prem_anchor_token = args
    if len(args)==3:
        current_prem_anchor_token, other_prem_anchor_token, is_negated = args

        # if "negated"
        if is_negated.data.strip() == 'neg':
            negated = True
            
    # Get dependent tokens on achor token for current and other premise
    current_prem_dep_tokens = easy_parse.get_dependent_tokens(premise.tokens, 
                                                              current_prem_anchor_token)
    other_prem_dep_tokens = easy_parse.get_dependent_tokens(other_premise.tokens, 
                                                            other_prem_anchor_token)

    # If current premise anchor is ROOT, remove all adverbial clause dependent tokens
    if current_prem_anchor_token.deprel == 'ROOT':
        
        # Get adverbial clause modifier
        advcls = [t for t in current_prem_dep_tokens if t.deprel == 'advcl']
        advcl_dep_tokens = []
        
        # Get dependent tokens on advcl
        for adv in advcls:
            advcl_dep_tokens += easy_parse.get_dependent_tokens(premise.tokens, adv)
            advcl_dep_tokens.append(adv)
            
        # Get list of all dependent tokens 
        advcl_dep_tokens = list(set(advcl_dep_tokens))
        
        # Exclude all these tokens for current premise
        current_prem_dep_tokens = [t for t in current_prem_dep_tokens if t not in advcl_dep_tokens] 
        
    # If other premise anchor is ROOT, remove all adverbial clause dependent tokens
    if other_prem_anchor_token.deprel == 'ROOT':
        
         # Get adverbial clause modifier
        advcls = [t for t in other_prem_dep_tokens if t.deprel == 'advcl']
        advcl_dep_tokens = []
        
        # Get dependent tokens on advcl
        for adv in advcls:
            advcl_dep_tokens += easy_parse.get_dependent_tokens(other_premise.tokens, adv)
            advcl_dep_tokens.append(adv)
            
        # Get list of all dependent tokens 
        advcl_dep_tokens = list(set(advcl_dep_tokens))
        
        # Exclude all these tokens for other premise
        other_prem_dep_tokens = [t for t in other_prem_dep_tokens if t not in advcl_dep_tokens] 
    
    # Append anchor token
    current_prem_dep_tokens.append(current_prem_anchor_token)
    other_prem_dep_tokens.append(other_prem_anchor_token)
        
    # Clean and sort lists by token id
    sorted_current_tokens = list(set(sorted(current_prem_dep_tokens, key=lambda token: token.id)))
    sorted_other_tokens = list(set(sorted(other_prem_dep_tokens, key=lambda token: token.id)))
            
    # Remove punctuations
    sorted_current_tokens = list(set([t.lemma for t in sorted_current_tokens if t.deprel not in ['mark','punct']]))
    sorted_other_tokens = list(set([t.lemma for t in sorted_other_tokens if t.deprel not in ['mark','punct']]))
    
    # If negated version, add a negation to non-negated sentence (for easy comparison)
    if negated:
        
        # Add "not" or "do not" to set that does not contain negation
        if 'not' not in sorted_current_tokens:
            if current_prem_anchor_token.lemma in ['be','can','must']:
                sorted_current_tokens.append('not')
            else:
                sorted_current_tokens += ['do','not']
                
        if 'not' not in sorted_other_tokens:
            if other_prem_anchor_token.lemma in ['be','can','must']:
                sorted_other_tokens.append('not')
            else:
                sorted_other_tokens += ['do','not']          
            
    # Sort lists
    sorted_current_tokens = sorted(sorted_current_tokens)
    sorted_other_tokens = sorted(sorted_other_tokens)
            
    # Return whether two token lists are identical
    return (sorted_current_tokens == sorted_other_tokens)

def exists(args, premise, verbose=True):
    """Check whether token with these keywords exists."""
    
    for token in premise.tokens:
        
        # Default: match, turns false as soon as mismatch in keyword is found
        all_match = True
        
        # Iterate through arguments...
        for arg in args:
            
            # Get keyword argument and value
            keyword = arg.keyword
            value = arg.kw_arg.data
            
            # Check whether it machces
            if getattr(token, keyword) != value:
                all_match = False
                
        # If every keyword matched
        if all_match:
            return True
        
    # If mismatch anywhere
    return False

def not_exists(args, premise, verbose=True):
    """Invert result for exist function."""
    
    # Get result for exist function with given arguments.
    result = exists(args, premise)
    
    # Invert
    if result:
        return False
    else:
        return True

def has_dep_token(args, premise, verbose=True):
    """Check whether token has dependent token with specific contraints."""
    
    # Get token and constraint
    token, constr = args

    # Get dependent tokens
    dependent_tokens = [t for t in premise.tokens if t.head == token.id]

    # Keyword arguments
    if constr.is_kwarg:

        # Lemma
        if constr.keyword == 'lemma':
            constr_value = constr.kw_arg.data
            
        # Deprel
        if constr.keyword == 'deprel':
            constr_value  = constr.kw_arg.data
        
        # For each dependent token    
        for t in dependent_tokens:

            # Check whether deprel is equal to given constraint
            if constr.keyword == 'deprel':                       
                if t.deprel == constr_value:
                    return True
            
            # Check whether lemma is equal to given constraint
            if constr.keyword == 'lemma':
                if t.lemma == constr_value:
                    return True

        # If no match is found, return False
        return False
                
    # If no keyword arugment is given
    else:
        
        # If constraint is "operators", check whether any dependent token is operator
        if constr.data == 'operators':
            for t in dependent_tokens:
                if t.lemma in rel_pol.operators:
                    return True                
        return False


def has_no_dep_token(args, premise, verbose=True):
    """Invert result from has_dep_token function."""
    
    # Get result for has_dep_token with given arguments
    result = has_dep_token(args, premise)

    # Invert result
    if result:
        inversed_result = False
    else:
        inversed_result = True

    return inversed_result


def words_in_sent(args, premise, verbose=True):
    """Check whether words (as string) are present in sentence."""

    # Get all words
    words = args[0].data.split(' ')

    # Check for each word whether it is contained in premise
    for word in words:
        if word not in premise.words:
            return False
        
    return True
    
def id_difference(args, premise, verbose=True):
    """Compute ID difference between two tokens and check wether it matches the
    given diff value."""
    
    # Get first, second tokend and diff value
    first_token, second_token, diff = args  
    
    # Get diff sign and value
    diff_sign = '-' if diff.sign == '-' else '+'
    diff_value = diff.value

    # If no sign, or positive:
    if diff_sign == '+' or diff_sign == None:
        is_equal = (first_token.id == second_token.id + diff_value)
    # If negative sign:
    else:
        is_equal = (first_token.id == second_token.id - diff_value)
        
    return is_equal
    
    
def dependent(args, premise, verbose=True):
    """Check whether first token is dependent on second token."""

    # Get tokens
    first_token, second_token = args
    
    # Compare first token head to second token ID
    if first_token.head == second_token.id:
        return True
    else:
        return False
       

def invert_boolean(args, premise, verbose=True):
    """Invert boolean value: True --> False, False --> True"""
    
    # Catch type error
    if not type(args[0]) == bool:
        raise TypeError('Cannot invert non-boolean value')
    
    # Invert
    if args[0] == True:
        reverse = False
    else:
        reverse = True

    return reverse
    
def exists_token(args, premise, verbose=True):
    """Introduce a token, always return True."""
    return True  
    
def token(args, premise):    
    """Return token."""
    return args[0]
        
def id(args, premise):
    """Return ID."""
    return args[0].id
    
def form(args, premise):
    "Return form (token as it shows up in sentence).""" 
    return args[0].form
    
def lemma(args, premise):
    """Return lemma."""
    return args[0].lemma
    
def u_pos(args, premise):    
    """Return universal POS-tag."""
    return args[0].u_pos
    
def x_pos(args, premise):    
    """Return POS-tag."""
    return args[0].x_pos
    
def head(args, premise):
    """Return head (ID of token current token depends on)."""
    return args[0].head
    
def deprel(args, premise):
    """Return dependency relation to head."""
    return args[0].deprel
    
def in_list(args, premise):
    """Check whether word exists in given list (list identified by name)."""
   
    # Get word and list name
    word, listname = args
    
    # Get list
    list_to_check = rel_pol.get_list_by_name(listname.data)
    
    # Check whether word exists in list
    if word in list_to_check:
        return True        
    else:
        return False

    
def context(args, premise, verbose=True):
    """Check whether string given by keyword arguments 'left' or 'right' matches
    context of reference token."""
    
    # Get keyword ID
    keyword_id = args[0].id 
    leftside = None
    rightside = None

    # Get keyword argument information
    for arg in args[1:]:
        if arg.is_kwarg:
        
            # "Left context"
            if arg.keyword == 'left':
                leftside = arg.kw_arg.data
                if leftside[0] in ['"', '"']:
                    leftside = leftside[1:-1]
                left_words = leftside.split(' ')
                
            # "Right context"
            if arg.keyword == 'right':
                rightside = arg.kw_arg.data
                if rightside[0] in ['"', '"']:
                    rightside = rightside[1:-1]
                right_words = rightside.split(' ')
                
    # If leftside context      
    if leftside:
    
        # If there are not enough words on left side, return False
        if keyword_id < len(left_words):
            return False
        
        # Invert word list
        left_words = left_words[::-1]

        # For each word, check whether word and position matches premise tokens
        for i, word in enumerate(left_words):
            if word.lower() != premise.tokens[keyword_id - (i+1)].form.lower():
                return False
            
    # If rightside context
    if rightside:
        
        # If there are not enough words left on the riht side, return False        
        if len(premise.tokens)-keyword_id < len(right_words):
            return False
        
        # For each word, check whether word and position matches premise tokens
        for i, word in enumerate(right_words):
            if word.lower() != premise.tokens[keyword_id + (i+1)].form.lower():
                return False                
    return True
    
                        
def prep_phrase(args, premise, verbose=True):
    """Check whether there is prepositional phrase in premise."""

    # Collect prepositions 
    preps = [t for t in premise.tokens if t.deprel == 'prep']
    
    if preps:
        return True
    else:
        return False
   
    
# ----------------------------------------
### TRANSITION FUNCTINS: insert, replace, delete, select
# ----------------------------------------    

def insert(arguments, premise, verbose=True):
    """Insert token into sentence at given position."""  
    
    # Default settings
    pos = None 
    correction_sign = '+'
    correction_value = 0
        
    # Get insertion token, orientation
    ins_token, orientation_token = arguments[:2]
    
    # Requires reparse when constant token is inserted (indicated by -1 ID)
    requires_reparse = (ins_token.id == -1)
    
    # For each argument...
    for arg in arguments[2:]:
            
        # If keyword argument
        if isinstance(arg, rrs.Argument) and arg.is_kwarg:
        
            # Get keyword and value
            keyword, kwarg = arg.keyword, arg.kw_arg
            
            # Get correction
            if keyword == 'correction':
            
                correction_sign = kwarg.sign
                correction_value = kwarg.value
                
            # Get POS
            if keyword == 'pos':
                pos = pos.kw_arg if isinstance(pos, rrs.Argument) and pos.is_kwarg else str(kwarg)
          
    # If POS is given, alter insertion token accordingly, and reparse
    if pos != None:   
        ins_token.form = get_inflected_word(ins_token, pos)
        requires_reparse = True
    
    # If it is an multi-premise rule, reparse
    if premise.is_mp_rule:
        requires_reparse = True
 
    # If orientation token is Argument
    if isinstance(orientation_token, rrs.Argument):
        
        # If keyword argument
        if orientation_token.is_kwarg:
        
            # Get position value
            if orientation_token.keyword == 'position':
                ins_position = orientation_token.kw_arg.value
   
    # If orientation token is string compute insertion position      
    else:
        # Get insertion position
        if correction_sign == '+' or correction_sign == None:
            ins_position = (orientation_token.id+correction_value)+1
        else:
            ins_position = (orientation_token.id-correction_value)+1  
            
    # Verbose
    if verbose:
        print('\nINSERT', ins_token.lemma)    
        print('Old premise:', " ".join(premise.words))

    # Increase positions for tokens after insertion
    for token in premise.tokens:
        if token.id>=ins_position:
            token.id += 1
        
    # Increase dependency positions for tokens with dependency higher than insertion position
    for token in premise.tokens:
        if token.head >=ins_position:
            token.head += 1
                    
    # Insert token at given position
    premise.tokens.insert(ins_position, ins_token) 
    
    # Get previous token
    prev_token = premise.tokens[ins_position-1]
    
    # If inserted token is "a", check whether it needs to be altered to "an"
    if ins_token.lemma == 'a' and ins_token.u_pos == 'DET':
        next_token = premise.tokens[ins_position+1]
        if next_token.u_pos in ['NOUN','ADJ'] and next_token.form[0] in ['a','e','i','o','u']:
            ins_token.form = 'an'
            
    # Change previous token "a" if next word starts with vowel
    if ins_token.u_pos in ['NOUN','ADJ'] and ins_token.form[0] in ['a','e','i','o','u']:
        if prev_token.lemma == 'a' and prev_token.u_pos == 'DET':
            prev_token.form = 'an'
    else:
        if prev_token.lemma == 'an' and prev_token.u_pos == 'DET':
            prev_token.form = 'a'
            
            
    # Adjust ID of inserted token
    try:
        premise.tokens[ins_position].id = ins_position
    except IndexError:
        premise.tokens[-1].id = ins_position
    
    # Reparse if necessary
    if requires_reparse:        
        premise = reparse(premise)
        
    # Update premise and set polarity scope             
    premise.update(premise)    
    
    if verbose:
        print('New premise:', " ".join(premise.words))
    
    # Check whether polarity settings could be computed
    successful_polarity_setting = premise.set_polarity_scope()
    
    # Get correct projectivity if possible
    if successful_polarity_setting:
        projectivity = rel_pol.projectivity_dict[ins_token.polarity]
    else:
        projectivity = None
        
    # Return new premise and projectivity
    return premise, projectivity    
    
def replace(arguments, premise, verbose=True):
    """Replace a token with a given other one."""

    # Get old token, new token, and POS (if given)
    if len(arguments)==3:
        old_token, new_token, pos = arguments
    else:
        old_token, new_token, pos = arguments + [None]
        
    # Get POS
    pos = pos.kw_arg if isinstance(pos, rrs.Argument) and pos.is_kwarg else pos 
    
    # --- GET RIGHT POSITION OF REPLACED WORD ---
    
    # 1) Get position of token to be replaced
    position = old_token.id
    
    # 2) Get position of lemma that matches lemma and deprel of token to be replaced 
    # (ID is not always reliable)
    for tok in premise.tokens:    
        if tok.lemma == old_token.lemma and tok.deprel == old_token.deprel:
            position = tok.id
            break
        
    # Get token to be replaced by position if possible
    try:
        repl_token = premise.tokens[position]
    except IndexError:
        return None, None

    # 3) If t has same attributes as old token, use that as token to be replaced 
    # and take that position
    for i,t in enumerate(premise.tokens):
        if t.same_token(old_token):
            repl_token = t
            position = i

    # Requires reparse when constant token is inserted (indicated by -1 ID)
    requires_reparse = repl_token.id == -1

    # Get required POS 
    required_pos = str(pos) if pos != None else repl_token.x_pos
    requires_reparse = (pos != None)

    # Reparse if rule is multi-premise rule
    if premise.is_mp_rule:
        requires_reparse = True
    
    # Get new token lemma and text
    if isinstance(new_token,str):
         new_token_text = get_inflected_word(new_token, required_pos)
         new_token_lemma = new_token
    else:
        new_token_text = get_inflected_word(new_token, required_pos)
        new_token_lemma = new_token.lemma
        
    # Printing
    if verbose:
        print('\nREPLACE', old_token.lemma, 'with', new_token_lemma)
        print('Old premise:', " ".join(premise.words))
        
    # Save new lemma and word
    repl_token.lemma = new_token_lemma
    repl_token.form  = new_token_text
    repl_token.x_pos = required_pos
    
    if repl_token.lemma == 'most':
        repl_token.form = new_token_lemma
        
    # Get previous token    
    prev_token = premise.tokens[repl_token.id-1]
    
    # Change determiner from "a" to "an" or backwards if necessary
    if repl_token.u_pos in ['NOUN','ADJ'] and repl_token.form[0] in ['a','e','i','o','u']:
        if prev_token.lemma == 'a' and prev_token.u_pos == 'DET':
            prev_token.form = 'an'
    else:
        if prev_token.lemma == 'an' and prev_token.u_pos == 'DET':
            prev_token.form = 'a'
        
    # Reparse if necessary
    if requires_reparse:
         premise = reparse(premise)
    
    # Update premise
    premise.update(premise)
    
    if verbose:
        print('New premise:', " ".join(premise.words))   
    
    # Set projectivity
    if new_token_lemma in ['some','all','every','no','many','few']:
        projectivity = 'OPERATOR'
    elif old_token.specific_projectivity == None:
        projectivity = rel_pol.projectivity_dict[old_token.polarity]
    else:
        projectivity = old_token.specific_projectivity
            
    # Return new premise and projectivity
    return premise, projectivity
    
                
def delete(arguments, premise, verbose=True):
    """Delete token from sentence."""
      
    # Get token to be deleted, and (if available) delete type
    if len(arguments)==2:
        del_token, del_type = arguments
        del_type = del_type.data
    else:
        del_token = arguments[0]
        del_type = None
        
    # Get deletion position
    if isinstance(del_token, str):
        for i, tok in enumerate(premise.tokens):
            if tok.lemma == del_token:
                del_token = tok
                break
            
    else:
        for i, tok in enumerate(premise.tokens):
                if tok.same_token(del_token):
                    del_token = tok

    
    # Printing               
    if verbose:
        print('\nDELETE', del_token.lemma)
        print('Old premise:', " ".join(premise.words))

    # Get all tokens that have to be delete
    del_tokens = [del_token]
    
    # If full phrase should be deleted                
    if del_type == 'full':    
        del_phrase = easy_parse.get_dependent_tokens(premise.tokens, del_token)
        del_tokens += del_phrase
            
    # Sort tokens to be deleted by ID
    del_tokens_sorted = sorted(del_tokens, key=lambda token: token.id)
                
    # All positions of tokens to be deleted
    del_tokens_positions = []
    
    # Save positions of tokens to be deleted
    for i,k in enumerate(premise.tokens):
        if k in del_tokens_sorted:
            del_tokens_positions.append(i)
                
    # Get first and last token
    last_del_token = max(del_tokens_positions)
    first_del_token = min(del_tokens_positions)
    
    # If relative clause, adjust positions (because of commas)
    if 'relcl' in [t.deprel for t in del_tokens_sorted]:    
        first_del_token -= 1
        last_del_token += 1
        
        del_tokens_positions.append(first_del_token)
        del_tokens_positions.append(last_del_token)
        
    # If necessary, increase position of last token to be deleted (e.g. punctuation)
    if len(premise.tokens) > last_del_token+1 and premise.tokens[last_del_token+1].lemma in [',',':',';']:
        del_tokens_positions.append(last_del_token+1)
        
    # Number of deleted tokens
    n_del_tokens = len(del_tokens_positions)
            
    # Get all tokens that are not supposed to be deleted
    new_premise_tokens = [t for i,t in enumerate(premise.tokens) if i not in del_tokens_positions]
    
    # Set position and dependency values for tokens
    for i,token in enumerate(new_premise_tokens):
        if i >= last_del_token:
            token.id = i - n_del_tokens
        if token.head >= last_del_token:
            token.head = token.head - n_del_tokens
                    
    # Save new premise tokens
    premise.tokens = new_premise_tokens
        
    # Reparse and update premise
    premise = reparse(premise)
    premise.update(premise)
    
    if verbose:
        print('New premise:', " ".join(premise.words))   

    # Get projectivity
    projectivity = rel_pol.projectivity_dict[del_token.polarity]
    
    # Return new premise and projectivity
    return premise, projectivity    


def select(arguments, premise, verbose=True):
    """Select token or phrase."""
    
    # Get token to be selected and selection type
    select_type = None
    select_token, select_type = arguments
    select_type = select_type.data

    # If full phrase should be deleted                
    if select_type == 'full':    
        select_tokens = easy_parse.get_dependent_tokens(premise.tokens, select_token)
        select_tokens.append(select_token)
        
    # Sort selected tokens by ID
    select_tokens_sorted = sorted(select_tokens, key=lambda token: token.id)
             
    # Ge new premise tokens
    new_premise_tokens = [t for t in select_tokens_sorted]
    premise.tokens = new_premise_tokens
        
    # Reparse and update premise
    premise = reparse(premise)
    premise.update(premise)
    
    if verbose:
        print('New premise:', " ".join(premise.words))   

    # Get projectivity
    projectivity = rel_pol.projectivity_dict[select_token.polarity]
    
    # Return new premise and projectivity
    return premise, projectivity  