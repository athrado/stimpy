# -*- coding: utf-8 -*-
"""
@author: jsuter
Project: STIMPY - Sentence Transformative Infernece Mapping for Python

Julia Suter, 2019
---
wordnet_relations.py

- Retrieve WordNet relations for given words or lists
- Recursively find more related words for extracted words
- Save results
"""

from nltk.corpus import wordnet as wn

import itertools
from itertools import chain
import pickle

# Dictionaries
        
pos_dict = {'NOUN':'n',
            'ADJ':'a',
            'ADV':'r',
            'VERB':'v'}
            
rev_pos_dict = {'n':'NOUN',
            'a':'ADJ',
            'r':'ADV',
            'v':'VERB'}
            
do_not_wordnet = [('be','v'),('will','v')]    


synonyms_dict = {}
hypernyms_dict = {} 
hyponyms_dict = {}
antonyms_dict = {}


def get_lowest_common_hypernym(word_pos_list_premise, word_pos_list_query, 
                               only_synsets_with_same_lemma =True, limit=10):
    """Get the lowest common hypernym of words in two lists."""
    
    additional_synsets = []    
    
    # For premise and query word pair
    for premise_w, query_w in itertools.product(word_pos_list_premise, word_pos_list_query):
        
            # Get premise word and POS
            word, pos = premise_w 
        
            # Add to pos dict
            if pos not in ['n','v','a','r']:
                pos = pos_dict[pos]          
                
            # Skip for stop words
            if (word, pos) in do_not_wordnet:
                continue
            
            # Get synysts for premise word
            premise_w_synsets = get_synsets(word, pos, only_synsets_with_same_lemma =True)     
            
            # Get query word and pos
            word, pos = query_w 
        
            # Add pos to dict
            if pos not in ['n','v','a','r']:
                pos = pos_dict[pos]          
                
            # Skip for stop words
            if (word, pos) in do_not_wordnet:
                continue
     
            # Get synsets for query word
            query_w_synsets = get_synsets(word, pos, only_synsets_with_same_lemma =True)     
                
            # Get lowest common hypernym
            for premise_synset, query_synset in itertools.product(premise_w_synsets, query_w_synsets):
            
                lowest_common_hypernym = premise_synset.lowest_common_hypernyms(query_synset)
                additional_synsets+=lowest_common_hypernym
                    
    # Return all additional synsets            
    return additional_synsets


def get_synsets(input_word, pos, only_synsets_with_same_lemma=True):
    """Get synsets."""
        
    # Get synsets for input word
    synsets = wn.synsets(input_word,pos=pos)
    
    # Filter for synsets that have same lemma
    if only_synsets_with_same_lemma:
        synsets = [syns for syns in synsets if syns.name().split('.')[0] == input_word]
    
    return synsets
    

def get_related_words(input_word, synsets, verbose=False, limit=10):
    """Get related words for input word and synsets."""
    
    # Prepare dicts
    all_synonyms = []
    all_hypernyms = []
    all_hyponyms = []
    all_antonyms = []
    
    # For each synset
    for i,j in enumerate(synsets):
        
        # Get all synonyms
        all_synonyms +=  [syn for syn in list(chain(*[j.lemma_names()])) if syn != input_word]
        # Get all hypernyms
        all_hypernyms += list(chain(*[l.lemma_names() for l in j.hypernyms()]))
        # Get all hyponyms
        all_hyponyms +=  list(chain(*[l.lemma_names() for l in j.hyponyms()]))
        # Get all antonyms
        all_antonyms +=  list(chain(*[[ant.name() for ant in l.antonyms()] for l in j.lemmas() if l.antonyms()]))

        # Print        
        if verbose:
            print("Meaning",i, "NLTK ID:", j.name())
            print('Definition:', j.definition())
            if len(j.examples())>0:
                print('Example:', j.examples()[0])
            print('\n')
    
            print("Synonyms:", ", ".join([syn for syn in list(chain(*[j.lemma_names()])) if syn != input_word]))
            print("Hypernyms:", ", ".join(list(chain(*[l.lemma_names() for l in j.hypernyms()]))))
            print("Hyponyms:", ", ".join(list(chain(*[l.lemma_names() for l in j.hyponyms()]))))
            print("Similar to:",  ", ".join(list(chain(*[l.lemma_names() for l in j.similar_tos()]))))
            print("Antonyms:", ", ".join(list(chain(*[[ant.name() for ant in l.antonyms()] for l in j.lemmas() if l.antonyms()]))))       
            print()
            
    # Return selection of extracted related words        
    return all_synonyms[:limit], all_hypernyms[:limit], all_hyponyms[:limit], all_antonyms[:limit]
        
def append_to_dict_value_list(dictionary, key, value, pos):
    """Append key,value and pos to dictionary."""
    
    # Get POS
    pos = rev_pos_dict[pos]
    word_pos = (key, pos)
    
    # Add to dictionary
    if word_pos in dictionary.keys():
        if value not in dictionary[word_pos]:
            dictionary[word_pos] += [value]
    else:
        dictionary[word_pos] = [value]


    
def add_wordnet_infos(word, pos, synonyms, hypernyms, hyponyms, antonyms, get_inverse):
    """Collect and save Wordnet information. """
    
    new_word_pos_list = []    
    
    # Synoynms
    for syn in synonyms:
    
        new_word_pos_list.append((syn, pos))        
        append_to_dict_value_list(synonyms_dict, word, syn, pos)
        
        if get_inverse:
            append_to_dict_value_list(synonyms_dict, syn, word, pos) 

    # Hypernyms
    for hyper in hypernyms:
    
        new_word_pos_list.append((hyper, pos))        
        append_to_dict_value_list(hypernyms_dict, word, hyper, pos)
        
        if get_inverse:
            append_to_dict_value_list(hyponyms_dict, hyper, word, pos)
        
    # Hyponyms
    for hypo in hyponyms:
    
        new_word_pos_list.append((hypo, pos))
        append_to_dict_value_list(hyponyms_dict, word, hypo, pos)
        
        if get_inverse:
            append_to_dict_value_list(hypernyms_dict, hypo, word, pos)                
    
    # Antonyms  
    for anto in antonyms:
        
        new_word_pos_list.append((anto,pos))        
        append_to_dict_value_list(antonyms_dict, word, anto, pos)
        
        if get_inverse:
            append_to_dict_value_list(antonyms_dict, anto, word, pos) 
            
    return new_word_pos_list


def get_all_wordnet_connections(word_pos_list_premise, word_pos_list_query=None, 
                                loops=1, save_dicts=True, get_inverse=False, 
                                get_common_hypernyms=True, verbose=False):
    """Recursively retrieve and save the WordNet relations for given list."""
    

    if get_common_hypernyms and word_pos_list_query != None:
        # Get additional synsets
        additional_synsets = get_lowest_common_hypernym(word_pos_list_premise, 
                                                        word_pos_list_query, 
                                                        only_synsets_with_same_lemma=True)
        # Word list for premise and query
        word_pos_list = word_pos_list_premise+word_pos_list_query
    else:
        word_pos_list = word_pos_list_premise
        
    # For each loop, get new related words
    for i in range(loops):
        
        new_word_pos_list = []
        
        # For word,pos pair
        for word_pos in word_pos_list:
            
            # Get word, pos
            word, pos = word_pos  

            if pos not in ['n','v','a','r']:
                pos = pos_dict[pos]          
                
            if (word, pos) in do_not_wordnet:
                continue
                
            # Get synsets
            synsets = get_synsets(word, pos, only_synsets_with_same_lemma =True)
            
            # Get additional synsets
            if i == 0 and get_common_hypernyms and word_pos_list_query != None:
                synsets += additional_synsets
                synsets = list(set(synsets))
                
            # Get related words
            synonyms, hypernyms, hyponyms, antonyms = get_related_words(word, synsets, limit=3)
            
            # Create the new word list
            new_word_pos_list += add_wordnet_infos(word, pos, synonyms, hypernyms, hyponyms, antonyms, get_inverse)
                        
        # New word list is processed further
        word_pos_list = [w for w in new_word_pos_list if w not in word_pos_list]
            
    # Save dicts
    if save_dicts:
        with open('../src/wordnet_dicts/synonyms_dict.pkl','wb') as fp:
            pickle.dump(synonyms_dict, fp)
            
        with open('../src/wordnet_dicts/hypernyms_dict.pkl','wb') as fp:
            pickle.dump(hypernyms_dict, fp)
            
        with open('../src//wordnet_dicts/hyponyms_dict.pkl','wb') as fp:
            pickle.dump(hyponyms_dict, fp)
            
        with open('../src//wordnet_dicts/antonyms_dict.pkl','wb') as fp:
            pickle.dump(antonyms_dict, fp)
            
            
    if verbose:
        
        print('Synonyms')
        print(synonyms_dict)
        print('\nHypernyms')
        print(hypernyms_dict)
        print('\nHyponyms')
        print(hyponyms_dict)
        print('\nAntonyms')
        print(antonyms_dict)
    
    return synonyms_dict, hypernyms_dict, hyponyms_dict, antonyms_dict    
    
 
def main():
    
    # Get WordNet dicts
    synonyms_dict, hypernyms_dict, hyponyms_dict, antonyms_dict = \
    get_all_wordnet_connections([('swede','NOUN'),('win','VERB'),
                                 ('prize','NOUN'),('literature','NOUN')])
    
    # Get synonyms
    for syn in synonyms_dict.keys():
        print(syn, synonyms_dict[syn])
    print()

    print('Synoyms')
    print(synonyms_dict)
    print()
    print('Hyponyms')
    print(hyponyms_dict)
    print()
    print('Hypernyms')
    print(hypernyms_dict)
    print()
    print('Antonyms')
    print(antonyms_dict)


if __name__ == "__main__":     
    main()
