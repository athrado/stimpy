# -*- coding: utf-8 -*-
"""
@author: jsuter
Project: STIMPY - Sentence Transformative Infernece Mapping for Python

Julia Suter, 2019
---
relations_and_polarity_settings.py

- Monotonicity dictionaries
- Projection dictionaries
- Operator information
"""


# ----------------------------------------
### LISTS
# ----------------------------------------

def get_list_by_name(listname):
    """Return list by name."""
    return listname_dict[listname]

non_affirmative_adjectives = ['former']
operators = ['all','most','many','some','several','few','no','both','every','each']

listname_dict = {'non_affirm_adj':non_affirmative_adjectives,
                 'operators':operators}

# ----------------------------------------
### RELATION AND MONOTONICITY DICTIONARIES
# ----------------------------------------

rel_as_symbols = {'forward entailment':'<',
                  'reverse entailment':'>',
                  'equivalence':'=',
                  'negation':'^',
                  'alternation':'|',
                  'cover':'~',
                  'unknown':'#'}


monotonicity_dict = {'all':('down','up'),
                     'every':('down','up'),
                     'some':('up','up'),
                     'not all':('up','down'),
                     'no':('down','down'),
                     'most':('non','up'),
                     'at least':('up','up'),
                     'less than':('non','down'),
                     'many':('non','non'),
                     'a few':('up','up'),
                     'few':('non','up'),
                     'the n':('non','up'),
                     'at most':('down','down'),
                     'half of the':('non','up'),
                     'exactly':('non','non'),
                     'neither':('down','down'),
                     
                     }
# ----------------------------------------
### PROJECTION DICTIONARIES
# ----------------------------------------

quantifier_projection_dict = {'some': ({'equivalence':'equivalence',
                        'forward entailment':'forward entailment',
                        'reverse entailment':'reverse entailment',
                        'negation':'cover',
                        'alternation':'unknown',
                        'cover':'cover',
                        'unknown':'unknown'},
                        
                        {'equivalence':'equivalence',
                        'forward entailment':'forward entailment',
                        'reverse entailment':'reverse entailment',
                        'negation':'cover',
                        'alternation':'unknown',
                        'cover':'cover',
                        'unknown':'unknown'}),
                        
                'no': ({'equivalence':'equivalence',
                        'forward entailment':'reverse entailment',
                        'reverse entailment':'forward entailment',
                        'negation':'alternation',
                        'alternation':'unknown',
                        'cover':'alternation',
                        'unknown':'unknown'},
                        
                        {'equivalence':'equivalence',
                        'forward entailment':'reverse entailment',
                        'reverse entailment':'forward entailment',
                        'negation':'alternation',
                        'alternation':'unknown',
                        'cover':'alternation',
                        'unknown':'unknown'})                      
                        
                        }
                        
                        
negation_projectivity_dict = {'reverse entailment':'forward entailment',
                   'forward entailment':'reverse entailment',
                   'negation':'negation',
                   'alternation':'cover',
                   'cover':'alternation',
                   'equivalence':'equivalence'}
                   
downward_projectivity_dict = {'reverse entailment':'forward entailment',
                   'forward entailment':'reverse entailment',
                   'negation':'negation',
                   'alternation':'alternation',
                   'cover':'cover',
                   'equivalence':'equivalence'}
                   
projection_dict = {'reverse entailment':'forward entailment',
                   'forward entailment':'reverse entailment',
                   'negation':'negation',
                   'alternation':'alternation',
                   'cover':'cover',
                   'equivalence':'equivalence'}
                   
upward_projectivity_dict = {'reverse entailment':'reverse entailment',
                   'forward entailment':'forward entailment',
                   'negation':'negation',
                   'alternation':'alternation',
                   'cover':'cover',
                   'unknown':'unknown',
                   'equivalence':'equivalence'}
                   
non_monotone_projectivity_dict = {'reverse entailment':'unknown',
                   'forward entailment':'unknown',
                   'negation':'negation',
                   'alternation':'alternation',
                   'cover':'cover',
                   'unknown':'unknown',
                   'equivalence':'equivalence'}
                   

projectivity_dict = {'up':upward_projectivity_dict,
                'down':downward_projectivity_dict,
                'negation':negation_projectivity_dict,
                'non': non_monotone_projectivity_dict}


# ----------------------------------------
### TRANSITION and VALIDITY dictionaries
# ----------------------------------------

transition_dict = { ('equivalence', 'reverse entailment'): 'reverse entailment',
                    ('equivalence', 'forward entailment'): 'forward entailment',
                    ('equivalence', 'negation'): 'negation',
                    ('equivalence','equivalence'):'equivalence',
                    ('equivalence', 'alternation'):'alternation',
                    ('equivalence', 'unknown'):'unknown',
                    ('equivalence', 'cover'):'cover',

                    ('forward entailment', 'forward  entailment'): 'forward entailment',
                    ('forward entailment', 'reverse entailment'): 'unknown',
                    ('forward entailment', 'negation'): 'alternation',
                    ('forward entailment', 'equivalence'):'forward entailment',
                    ('forward entailment', 'unknown'):'unknown',
                    ('forward entailment', 'cover'):'unknown',
                    ('forward entailment', 'forward entailment'):'forward entailment',
                    ('forward entailment', 'alternation'):'alternation',

                    ('reverse entailment', 'equivalence'):'reverse entailment',
                    ('reverse entailment', 'reverse entailment'): 'reverse entailment',
                    ('reverse entailment', 'forward entailment'): 'unknown',
                    ('reverse entailment', 'negation'): 'cover',
                    ('reverse entailment', 'cover'): 'cover',
                    ('reverse entailment', 'alternation'):'unknown',
                   
                    ('negation', 'reverse entailment'): 'alternation',
                    ('negation', 'forward entailment'): 'cover',
                    ('negation', 'negation'): 'equivalence',
                    ('negation','equivalence'):'negation',
                    ('negation', 'cover'):'forward entailment',
                    ('negation', 'alternation'):'reverse entailment',
                    ('negation', 'unknown'):'unknown',
                   
                    ('alternation', 'forward entailment'):'unknown',
                    ('alternation','alternation'):'unknown',
                    ('alternation', 'unknown'):'unknown',
                    ('alternation', 'cover'):'forward entailment',
                    ('alternation', 'reverse entailment'):'alternation',
                    ('alternation', 'negation'): 'forward entailment',
                    ('alternation', 'equivalence'):'alternation',
                   
                    ('cover', 'cover'):'unknown',
                    ('cover', 'negation'):'reverse entailment',
                    ('cover','forward entailment'):'cover',
                    ('cover', 'reverse entailment'):'unknown',

                    ('unknown', 'unknown'):'unknown',
                    ('unknown', 'reverse entailment'):'unknown',
                    ('unknown', 'forward entailment'):'unknown',
                    ('unknown', 'negation'):'unknown',
                    ('unknown', 'equivalence'):'unknown',
                    ('unknown','alternation'):'unknown',
                    ('unknown', 'cover'):'unknown',
                    
                    ('equivalence', None): 'equivalence',
                    ('forward entailment', None): 'forward entailment',
                    ('reverse entailment', None): 'reverse entailment',
                    ('negation', None): 'negation',
                    ('alternation', None): 'negation',
                    ('cover', None): 'cover',
                    ('unknown', None):'unknown',
                  }
                  
                  
validity_state_dict = {('valid', 'equivalence'):'valid',
                  ('valid', 'forward entailment'):'valid',
                  ('valid', 'alternation'):'invalid',
                  ('valid', 'negation'):'invalid',
                  ('valid', 'reverse entailment'):'unknown',
                  ('valid', 'cover'):'unknown',
                  ('valid', 'unknown'):'unknown',

                  ('invalid', 'cover'):'valid',
                  ('invalid', 'negation'):'valid',
                  ('invalid', 'equivalence'):'invalid',
                  ('invalid', 'reverse entailment'):'invalid',
                  ('invalid', 'alternation'):'unknown',
                  ('invalid', 'forward entailment'):'unknown',

                  ('unknown', 'alternation'):'unknown',
                  ('unknown', 'forward entailment'):'unknown',
                  ('unknown', 'reverse entailment'):'unknown',
                  ('unknown', 'cover'):'unknown',
                  ('unknown', 'negation'):'unknown',
                  ('unknown', 'equivalence'):'unknown',
                  ('unknown', 'unknown'):'unknown',
                  }
            