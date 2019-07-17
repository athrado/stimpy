# -*- coding: utf-8 -*-
"""
@author: jsuter
Project: STIMPY - Sentence Transformative Infernece Mapping for Python

Julia Suter, 2019
---
rule_settings.py

- System settings, e.g. whether or not to include WordNet information
- Token and predicate dictionaries
- Rule name dictionaries
- Load rule sets
"""

# ----------------------------------------
### Imports
# ----------------------------------------
import functions
import parse_information as easy_parse

# ----------------------------------------
### SETTINGS
# ----------------------------------------

include_wordnet = True

# ----------------------------------------
### DICTIONARIES
# ----------------------------------------

# Constant Token Dictionary
token_dict = {'NOT_' : easy_parse.Token([-1, 'not', 'not', 'ADV', 'RB', 1, 'neg']),
             'NO_' : easy_parse.Token([-1, 'No', 'no', 'DET', 'DT', 1, 'det']),
             'DO_' : easy_parse.Token([-1,'do','do','VERB','VBX',1,'aux']),
            'A_': easy_parse.Token([-1,'a','a','DET','DT',1,'det']),
            'THE_': easy_parse.Token([-1,'The','the','DET','DT',1,'det']),
            'THERE_': easy_parse.Token([-1,'There','there','DET','DT',1,'det']),
            'BE_': easy_parse.Token([-1,'be','be','VERB','AUX',1,'verb']),
            'THAT_':easy_parse.Token([-1,'that','that','ADJ','WDT',1,'nsubj']),
            'WHO_':easy_parse.Token([-1,'who','who','ADJ','WDT',1,'nsubj']),
            'ALL_': easy_parse.Token([-1,'all','all','DET','DT',2,'det']),
            'SOME_': easy_parse.Token([-1, 'Some','some', 'DET', 'DT',	2,'det']),
            'MANY_':easy_parse.Token([-1, 'many', 'many', 'ADJ','JJ',2,'amod']),
            'MOST_':easy_parse.Token([-1,'most',	'most','ADJ','JJ',1,'amod']),
            'EACH_':easy_parse.Token([-1,'Each','each','DET',	'DT',	2, 'det']),
            'FEW_':easy_parse.Token([-1,'Few','few','ADJ',	'JJ',	2, 'amod']),
            'EVERY_':easy_parse.Token([-1,'Every','every','DET',	'DT',	2, 'det']),
            'BOTH_':easy_parse.Token([-1,'both','both','DET','DT',1,'det']),
            'IS_': easy_parse.Token([-1, 'is','be','VERB','VBZ',1,'ROOT']),
            'SEVERAL_': easy_parse.Token([-1,'several','several','ADJ','JJ',	1,'amod']),
            'PLACEHOLDER_': easy_parse.Token([-1,'_','_','_','_',1,'_']),            
            }

# Dict for predicate variable limitations/constraints
pred_variable_limitations = {
                  'hypernym':[['NOUN','VERB','ADJ','XX']],
                  'hyponym':[['NOUN','VERB']],
                  'sibling':[['NOUN','VERB','ADJ']],
                  'synonym':[['NOUN','VERB','ADJ']],
                  'antonym':[],
                  'exists_hypernym':[['NOUN','VERB','ADJ']],
                  'exists_hyponym':[['NOUN','VERB','ADJ']],
                  'exists_sibling':[['NOUN','VERB','ADJ']],
                  'exists_synonym':[['NOUN','VERB','ADJ']],
                  'exists_antonym':[],
                  'positive_sent':[],
                  'not':[],
                  'prep_phrase':[],
                  'is':[],
                  'token':[],
                  'lemma':[],
                  'x_pos':[],
                  'id':[],
                  'id_difference':[],
                  'head':[],
                  'dependent':[],         
                  'in_list':[],
                  'context':[],     
                  'pluralize':[],
                  'compare_operators':[],
                  'singularize':[],
                  'deprel':[],
                  'is_passive': [],
                  'number_adjustment':[],
                  'same_phrase':[],
                  'same_phrase_negated':[],
                  'words_in_sent':[],
                  'has_dep_token':[],
                  'has_no_dep_token':[],
                  'exists':[],
                  'not_exists':[],
                                 
                 }               

# Predicate dictionary
predicate_dict = {'exists_hypernym':functions.exists_hypernym,
                  'exists_hyponym':functions.exists_hyponym,
                  'exists_sibling':functions.exists_sibling,
                  'exists_synonym':functions.exists_synonym,
                  'exists_antonym':functions.exists_antonym,
                  'synonym':functions.get_synonym,
                  'hypernym':functions.get_hypernym,
                  'hyponym':functions.get_hyponym,
                  'sibling':functions.get_sibling,
                  'antonym':functions.get_antonym,
                  'positive_sent':functions.positive_sent,
                  'not':functions.invert_boolean,
                  'prep_phrase':functions.prep_phrase,
                  'is':functions.exists_token,
                  'token':functions.token,
                  'id':functions.id,
                  'lemma':functions.lemma,
                  'form':functions.form,
                  'u_pos':functions.u_pos,
                  'x_pos':functions.x_pos,
                  'deprel':functions.deprel,
                  'head':functions.head,
                  'id_difference': functions.id_difference,
                  'dependent':functions.dependent,
                  'in_list':functions.in_list,
                  'context':functions.context,
                  'singularize':functions.singularize,
                  'pluralize':functions.pluralize,
                  'compare_operators':functions.operator_comparison,
                  'is_passive':functions.is_passive,
                  'number_adjustment':functions.operator_replacement_action,
                  'same_phrase':functions.same_phrase,
                  'same_phrase_negated':functions.same_phrase_negated,
                  'words_in_sent':functions.words_in_sent,
                  'has_dep_token':functions.has_dep_token,
                  'has_no_dep_token':functions.has_no_dep_token,
                  'exists':functions.exists,
                  'not_exists':functions.not_exists,
                  }
  
# Transition operation/function dictionary                
trans_operation_dict = {'insert':functions.insert,
                        'replace':functions.replace,
                        'delete':functions.delete,
                        'select':functions.select,                        
                        }
                        


# ----------------------------------------
### RULE SETS by rule name
# ----------------------------------------

# Rule can be applied several time on same token set
on_same_token_rules = [ "exists_hypernym(X) --> replace(X, hypernym(X)) # relation = forward entailment",
           "exists_hyponym(X) --> replace(X, hyponym(X)) # relation =  reverse entailment",
           "exists_sibling(X) --> replace(X, sibling(X)) # relation = alternation",
           "exists_synonym(X) --> replace(X, synonym(X)) # relation = equivalence",
           "exists_antonym(X) --> replace(X, antonym(X)) # relation = alternation"]
           
           
# Wordnet rules
wordnet_rules =   ["exists_hypernym(X) --> replace(X, hypernym(X)) # relation = forward entailment",
           "exists_hyponym(X) --> replace(X, hyponym(X)) # relation =  reverse entailment",
           "exists_sibling(X) --> replace(X, sibling(X)) # relation = alternation",
           "exists_synonym(X) --> replace(X, synonym(X)) # relation = equivalence",
           "exists_antonym(X) --> replace(X, antonym(X)) # relation = alternation"]        
           
# Rules with limited applications (no not necessarly repeat for every token list)            
rules_with_limied_applition = ['Replace with equal operator without adaptation',
                               'Replace with weaker operator without adaptation',
                               'Replace with stroger operator without adaptation',
                               
                               'Replace with equal operator with singularization',
                               'Replace with weaker operator with singularization',
                               'Replace with stronger operator with singularization',
                              
                               'Replace with equal operator with pluralization',
                               'Replace with weaker operator with pluralization',
                               'Replace with stronger operator with pluralization',
                               
                               'Replace with equal operator without adaptation II',
                               'Replace with weaker operator without adaptation II',
                               'Replace with stroger operator without adaptation II',
                               
                               'Replace with equal operator with singularization II',
                               'Replace with weaker operator with singularization II',
                               'Replace with stronger operator with singularization II',
                              
                               'Replace with equal operator with pluralization II',
                               'Replace with weaker operator with pluralization II',
                               'Replace with stronger operator with pluralization II',

                               'Sibling',
                               'Synonym',
                               'Antonym'                                   
                               ]

# ----------------------------------------
### LOAD RULES from file
# ----------------------------------------

# Open file
with open("../Rules/rules.txt") as infile:

    # Default settings
    rule_set  = []
    has_name = False
    
    # Read file
    rule_set_raw  = infile.read()

    # Split at linebreaks
    for x in rule_set_raw.split('\n'):
        
        # Ignore lines starting with # or ticks
        if x.startswith('#')  or x.startswith('"') or x.strip() == '':
            continue
    
        # Remove tabs
        elif x.startswith('\t'):
            rule_set[-1]+=x[1:]
            
        # Get rule names
        elif x.startswith('~'):
            rule_set.append(x)
            has_name = True
        else:
            if has_name:
                rule_set[-1]+='\n'+x
                has_name = False
            else:
                rule_set.append(x)
                
# Open file
with open("../Rules/mp_rules.txt") as infile:

    mp_rule_set  = []
    
    # Read file
    rule_set_raw  = infile.read()
    
    # Split at double linebreak
    for x in rule_set_raw.split('\n\n'):
        mp_rule_set.append(x.strip())