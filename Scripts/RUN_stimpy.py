# -*- coding: utf-8 -*-
"""
@author: jsuter
Project: STIMPY - Sentence Transformative Infernece Mapping for Python

Julia Suter, 2019
---
RUN_stimpy.py

Pipeline that runs Sentence Transformative Inference Mapping (STIMPY) engine
- Start building of inference tree
- Get new derived facts by evaluating conditions and applying transitions
- Monitor and save transitions
- Catch exceptions and errors
- Terminate when solution for infernece problem is found
- Return results in form of inference tree, computed relation and validity
"""

# ----------------------------------------
### Import Statements
# ----------------------------------------

import copy
import itertools
import pickle
import warnings

import rule_settings 
import relations_and_polarity_settings as rel_pol
import rule_reading_system as rrs
import parse_information as easy_parse

from anytree import Node, RenderTree
from anytree.exporter import JsonExporter
exporter = JsonExporter(indent=2, sort_keys=True)

import spacy

# Settings
import warnings
warnings.filterwarnings("ignore")

# Get spacy nlp for English
nlp = spacy.load('en')

# ----------------------------------------
### CLASSES for Query, Premise etc.
# ----------------------------------------

class Query():
    """Class for representing Query."""
    
    def __init__(self, tokens, sent):
        
        # Initialize Query attributes
        self.tokens = tokens
        self.sent = sent
        self.words = [t.form for t in tokens]
        self.words_lower = [t.lower() for t in self.words]
        self.lemmas = [t.lemma for t in tokens]
        self.string = " ".join(self.words)
        
       
    def __str__(self):
        """Return string for Query."""
        
        all_tokens = []
        
        # For each token
        for token in self.tokens:       
            
            # Get all attributes and join
            token_list = [str(token.id), token.form, token.lemma, \
                          token.u_pos, token.x_pos, str(token.head), \
                          token.deprel, token.polarity]
            all_tokens.append("\t".join(token_list))

        # Return as string
        return "\n".join(all_tokens)+'\n'
    
    def match(self, test_premise):
        """Check whether query matches given premise"""

        # If all query words equal to premise words
        is_match = (self.words_lower == test_premise.words_lower)
           
        return is_match
        

class Premise():
    """Class for represneting Premise."""
    
    def __init__(self, tokens, sent, state='equivalence', validity='valid', 
                 other_premises=None):
        
        # Initialize Premise attributes
        self.tokens = tokens
        self.sent = sent
        self.words = [t.form for t in tokens]
        self.lemmas = [t.lemma for t in tokens]
        
        # Length and string
        self.length = len(self.tokens)
        self.string = " ".join(self.words)
        
        # State and validity
        self.state = state
        self.validity = validity
        
        self.is_mp_rule = False      
        self.skip_n_rules = 0        
        self.TRANS = None        
        self.var_dict = None
        
        self.original_premise = None
        self.limited_premises = []
        
        # Other premises        
        if other_premises != None:
            self.other_premises = other_premises
        

    def __str__(self):
        """Return Premise as string."""
        
        all_tokens = []
        
        # For each token
        for token in self.tokens:       
            
            # Get all attributes and join
            token_list = [str(token.id), token.form, token.lemma, \
                          token.u_pos, token.x_pos, str(token.head), \
                          token.deprel, token.polarity]
            all_tokens.append("\t".join(token_list))

        # Return as string
        return "\n".join(all_tokens)+'\n'
         
    def premise_sent(self):
        """Return premise sentence as string."""
        
        # Return premise as sentence string
        return " ".join([t.form.lower() for t in self.tokens])
            
                
    def update(self, premise):
        """Update premise and its attributes."""
        
        self.tokens = premise.tokens
        self.sent = premise.sent
        self.words = [t.form for t in self.tokens]
        self.words_lower = [t.lower() for t in self.words]
        self.lemmas = [t.lemma for t in self.tokens]
        
        # Set new polarity scope
        self.set_polarity_scope()
        
        
    def set_polarity_scope(self):
        "Set polarity scope of quantifiers."
        
        # Set all token polarities to "up" by default
        for token in self.tokens:            
            token.polarity = 'up'
            token.specific_projectiviy = None
        
        # Fore each token, get polarity
        for token in self.tokens:
                                 
            # If token is quantifier
            if token.lemma in rel_pol.monotonicity_dict.keys() \
                              and token.deprel == 'det':
                           
                # Get first argument (noun phrase quantifier refers to)
                arg_1 = next(t for t in self.tokens if token.head == t.id)
                arg_1 = [t for t in easy_parse.get_dependent_tokens(
                                               self.tokens, arg_1)]+[arg_1]
                
                # Get second argument (VP)
                arg_2 = next(t for t in self.tokens if t.deprel == 'ROOT')
                arg_2 = [t for t in easy_parse.get_dependent_tokens(
                                        self.tokens, arg_2) if t not in arg_1]
            
                # Set polarity (up/down/non) for tokens in scope 
                for t in arg_1:
                    t.polarity = get_new_polarity(t.polarity, 
                                                  rel_pol.monotonicity_dict[token.lemma][0])

                for t in arg_2:   
                    t.polarity = get_new_polarity(t.polarity, 
                                                  rel_pol.monotonicity_dict[token.lemma][1])
                
                # If there is a specific projection for this quantifier, set it
                if token.lemma in rel_pol.quantifier_projection_dict.keys():
                    
                    # First argument of operator
                    for t in arg_1:
                        t.specific_projectivity = \
                        rel_pol.quantifier_projection_dict[token.lemma][0]
                        
                    # Second argument of operator
                    for t in arg_2:
                        t.specific_projectivity = \
                        rel_pol.quantifier_projection_dict[token.lemma][1]
                        
            # If negation
            if token.lemma == 'not':
                
                # Get root
                try:
                    root = next(t for t in self.tokens if t.deprel 
                                in ['ROOT'] and t.id == token.head)
                # Excpetion
                except StopIteration:
                    warnings.warn('Root not found')
                    return False
                    
                # Get subj
                try:
                    subj = next(t for t in self.tokens if t.deprel 
                                in ['nsubj','nsubjpass'] and t.head == root.id)
                # Exception
                except StopIteration:
                    warnings.warn('Root not found')
                    return False
                
                # Get full subject and VP
                full_subj =  [t for t in 
                              easy_parse.get_dependent_tokens(self.tokens, subj)]
                VP  =  [t for t in easy_parse.get_dependent_tokens(self.tokens, root) \
                        if t not in full_subj]
                
                # Set downward polarity to tokens in VP
                for t in VP:
                    t.specific_projectivity = rel_pol.negation_projectivity_dict 
                    t.polarity = 'down'
              
        return True                
      
    def all_upward_monotone(self):
          """Check if all tokens have upward polarity."""
          
          # For each token, check whether polarity is "up"
          for token in self.tokens:
              if token.polarity != 'up':
                  return False
         
          return True
          
                    
class Transformation_Settings():
    """Class for Inference Tracking."""
    
    def __init__(self, premise, query):        
        
        # Save original premise
        self.original_premise = copy.deepcopy(premise)
        
        # Save all correct permutations (as premise and words)
        self.permutations_premise = [premise.original_premise]
        self.permutations_words = [premise.words]
        self.all_branches = [" ".join(premise.words)]
        self.all_applied_rules = []
        
        # Save node
        self.root_node =  Node(" ".join(premise.words))
        
        # Match found or not
        self.found_match = False
        
        # Save query
        self.query = query
        self.limited_premises = []    
        
        # Initialize lists for supervising transformation steps
        self.processed_hypernym_pairs = []
        self.processed_hyponym_pairs = []
        self.processed_sibling_pairs = []
        self.processed_synonym_pairs = []
        self.processed_antonym_pairs = []
        
        # Unknown settings
        self.final_relation = 'UNKNOWN'
        self.final_validity = 'UNKNOWN'
    
        self.fallback_relation = 'unknown'
        
        
    def reset_wordnet(self):
        
        self.available_hyponyms = 0
        self.available_hypernyms = 0
        self.available_synonyms = 0
        self.available_antonyms = 0
        self.available_siblings = 0
        
# ----------------------------------------
### Functions for preparing sentences and tokens, setting polarity,
### printing inference tree          
# ----------------------------------------

def get_tokens_and_sent(sent):
    """Return token and sent as class objects."""

    # Set counters
    word_pos_counter = sent[0].i
    sentence = []
    space_chars = 0

    # For each sent
    for i, word in enumerate(sent):

        # If word is simply space, skip it 
        if word.pos_ == 'SPACE':
            space_chars += 1
            continue

        # Collect parsing infos 
        parse_info = [
            word.i-word_pos_counter-space_chars,  # adapt position if there are white characters
            word.text,
            word.lemma_,
            word.pos_, 
            word.tag_, 
            word.head.i-word_pos_counter-space_chars, # adapt position if there are white characters
            word.dep_]

        sentence.append(parse_info)

    # Transform into Token and Sentence object
    tokens = [easy_parse.Token(k) for k in sentence]  
    sent = easy_parse.Sentence(tokens)

    return tokens, sent
    
def get_new_polarity(original_pol, monotoncity):
    "Determine new polarity for this lexical item."
    
    # If original polarity is upward, return new monotonicity
    if original_pol == 'up':
        return monotoncity
    
    # If original polarity is not upward, invert
    else:
        if monotoncity == 'up':
            return 'down'
        else:
            return 'up'
    
def print_inference_tree(root_node):
    """Print inference tree."""
    
    # Import and save
    from anytree.exporter import DotExporter
    DotExporter(root_node).to_picture("results/transformation_tree.png")
    
    # Print
    for pre, fill, node in RenderTree(root_node):
        print("%s%s" % (pre, node.name))
    
# ----------------------------------------
### TRANSFORMATION pipeline          
# ----------------------------------------

def start_transformation_pipeline(rules, premise, query, verbose=False, full_tree=False):
    """Pipeline for tranformation."""
    
    # Get Inference Tracking started
    TRANS = Transformation_Settings(premise, query)
    
    # Set root node
    root_node = Node(" ".join(premise.words))           
    
    # Set depth
    depth = 1
              
    # Start processing inference premises
    process_premise_sets([(premise, (rules, query, TRANS, root_node))],
                       depth, verbose=verbose, full_tree=full_tree)
    
    return root_node, TRANS


def process_premise_sets(premises, depth, verbose=False, full_tree=False):
    """Process every premise in premise set 
    and feed resulting new premises back into system."""
    
    
    # Check depth level and continue
    if depth > 3:
        print("Depth "+str(depth)+"\nYou've searched far enough...")
        return False

    all_new_premises = []            
    
    # For each premise
    for premise in premises:
                          
        # Get premise and arugments
        premise, arguments = premise
        
        # Get rules and query
        rules, query, TRANS, root_node = arguments

        # Get new premises (by transitions)
        new_premises = check_conditions(premise, arguments, verbose=verbose, full_tree=full_tree)
        
        # If inference transformations were successfull
        if new_premises == True:   
            
            # Save tree
            with open ('results/anytree_intermediate.json', 'w', encoding = 'utf-8') as f:
                exporter.write(root_node,f)
            return True
        
        # If inference transtiomations were not successfull
        elif new_premises == False:
            continue
        
        
        # If there are more premises to process
        else:            
            # For each new premise
            for new_prem in new_premises:
                
                # Save node and tree
                tree_node = Node(" ".join(new_prem.words), parent=root_node, edge=new_prem.prev_rule)
                if full_tree:
                    with open ('results/anytree_intermediate.json', 'w', encoding = 'utf-8') as f:
                        exporter.write(root_node,f)
                with open('results/all_branches_and_rules.pkl','wb') as f: 
                   pickle.dump((TRANS.all_branches, TRANS.all_applied_rules), f)
                     
                # Get arguments
                arguments = rules, query, TRANS, tree_node
                # Save new premises
                all_new_premises.append((new_prem, (arguments)))
            
    # If there are new premises to process
    if all_new_premises:  
        process_premise_sets(all_new_premises, depth+1, verbose=verbose, full_tree=full_tree)
    # Otherwise, stop
    else:
        return False


def check_conditions(prem, arguments, verbose=False, mp_rules=True, full_tree=False):
    """Check variables, conditions, other premises
    and send to rule application."""
    
    # Get arguments
    rules, query, TRANS, root_node = arguments
    
    new_premises = []
    
    # Get deep copy of premise (copy can be changed)
    premise = copy.deepcopy(prem)
         
    # For each rule
    for i,rule in enumerate(rules):
        
        # For single premise rules
        if not rule.multipremise_rule:
            
            premise.is_mp_rule = False
    
            # For rules with limitations
            if rule.rule_name in rule_settings.rules_with_limied_applition: 
                
                # If premise and rule combination was already tested, continue
                if (rule.rule_name, premise.premise_sent()) in TRANS.limited_premises:
                    
                    # Save 
                    TRANS.limited_premises.append(
                            (rule.rule_name, premise.premise_sent()))
                    warnings.warn('This premise/rule combination was already tested')
    
                    continue
           
            # If not activated, do not process wordnet rules
            if not rule_settings.include_wordnet \
                and rule.data in rule_settings.wordnet_rules:
                    continue
                
            # Get copy of premise tokens
            premise_tokens = copy.deepcopy(premise.tokens)
            
            # Check whether there are exclusive variables that match token
            exclusive_variables_checked = rrs.check_exclusive_variables(
                    premise_tokens, rule.exclusive_var_dict)
            
            # Continue if exclusive variables are found
            if exclusive_variables_checked:
                continue
            
            # Get all token permutations for (normal) variables                                                          
            tokens_for_variables = itertools.permutations(premise_tokens, rule.n_var)                
            tokens_for_variables = list(tokens_for_variables)
                                  
            # For each set
            for token_set in tokens_for_variables:
                
                # If tokens fit variables
                if rrs.matches_constraints(token_set, rule.var_dict):
                                                                
                    # Get the variable dict
                    premise.var_dict = rule.var_dict
                    
                    # Save tokens in var dict
                    token_var_dict = dict(zip(rule.var_dict, token_set)) 
     
                    # Save TRANS and resert wordnet info                       
                    premise.TRANS = TRANS                        
                    premise.TRANS.reset_wordnet()

                    # If there is a list of constant variables
                    if rule.const_variables_list:
                        
                        # For each constant in list, try rule with that constant as variable                       
                        for x in list(itertools.product(*rule.const_variables_list.values())):
                            new_dict = dict(zip(rule.const_variables_list.keys(), x))
                            
                            # For each element
                            for elem in new_dict:
                                
                                # Get constant
                                const = new_dict[elem]                                 
                                # Save value in token var dict
                                token_var_dict[elem] = copy.deepcopy(rule_settings.token_dict[const])
                                                                
                            # Apply transitions and get new premises
                            new_premises, premise, root_node, finished = apply_transitions(rule, premise, TRANS, query, token_var_dict, new_premises, prem, root_node, full_tree=full_tree)
                            
                            # If rule should only be applied in limited fashion,
                            # save that premise/rule combination
                            if rule.rule_name in rule_settings.rules_with_limied_applition:  
                                 
                                # For each premise
                                 for p in new_premises:
                                     TRANS.limited_premises.append((rule.rule_name, p.premise_sent()))     
                                
                            # Stop processing if finished
                            if finished:
                                return True
          
                    # If there are no list of constant variables
                    else:
                            # Apply transitions and get new premises
                            new_premises, premise, root_node, finished = apply_transitions(rule, premise, TRANS, query, token_var_dict, new_premises, prem, root_node, full_tree=full_tree)                    
                                                        
                            # If rule should only be applied in limited fashion,
                            # save that premise/rule combination
                            if rule.rule_name in rule_settings.rules_with_limied_applition:  
                                
                                # For each premise 
                                for p in new_premises:
                                     TRANS.limited_premises.append((rule.rule_name, p.premise_sent()))   
                            
                            # Stop processing if finished
                            if finished:
                                return True                       

                

        # For multi-premise rules
        else:
            
            premise.is_mp_rule = True
            
            # Only proceed if there are other premises
            if premise.other_premises:
            
                # For rules with limitations
                if rule.rule_name in rule_settings.rules_with_limied_applition:    
                    
                    # If premise and rule combination was already tested, continue
                    if (rule.rule_name, premise.premise_sent()) in TRANS.limited_premises:
                        
                        # Save 
                        TRANS.limited_premises.append((rule.rule_name, premise.premise_sent()))
                        warnings.warn('This premise/rule combination was already tested')
                        
                        continue
               
                # If not activated, do not process wordnet rules
                if not rule_settings.include_wordnet and rule.data in rule_settings.wordnet_rules:
                    continue
                    
                # Get a copy of premise tokens
                premise_tokens = copy.deepcopy(premise.tokens)
                
                # Get variable information
                nr_var_premise = len(rule.var_dict_per_premise[1])
                first_premise_var_dict = rule.var_dict_per_premise[1]
                nr_other_premises = len(rule.var_dict_per_premise)-1
         
                # Get all token permutations for variables                                                          
                tokens_for_variables = itertools.permutations(premise_tokens, nr_var_premise)                
                tokens_for_variables = list(tokens_for_variables)
                
                # For each set
                for token_set in tokens_for_variables:
                    
                    # If tokens fit variables
                    if rrs.matches_constraints(token_set, first_premise_var_dict):
                        
                        # Get the variable dict
                        premise.var_dict = first_premise_var_dict                            
                                                    
                        # Save tokens in var dict
                        token_var_dict = dict(zip(first_premise_var_dict, token_set)) 
                        
                        # Create final variable dict
                        final_var_dict = {}
                        final_var_dict = {**final_var_dict, **token_var_dict}
                        
                        # Save TRANS and resert wordnet info                       
                        premise.TRANS = TRANS                        
                        premise.TRANS.reset_wordnet()                                
                        
                        # Get all rule elements (seperately)
                        elements = rule.condition_sets[0].elements   
        
                        # Evalute elements to see if it can be successfully evaluated
                        successfully_evaluated = rrs.evaluate_cond_elements(elements, premise, token_var_dict)
                   
                        # If current premise was successfully evaluted, 
                        # check whether other premises match
                        if successfully_evaluated:
                                           
                            # For other premises
                            for j in range(nr_other_premises):
                                
                                j += 2
                                
                                # Get correct variable dict
                                rule.var_dict_per_premise[j]
                                    
                                # For each other premise, check whether it fullfills conditions
                                for other_prem in premise.other_premises:
                            
                                    # Prepare premise and tokens
                                    premise.other_premise = other_prem
                                    other_premise_tokens = copy.deepcopy(other_prem.tokens)
                                    
                                    # Get all token permutations for variables                                                          
                                    tokens_for_variables = itertools.permutations(other_premise_tokens, len(rule.var_dict_per_premise[j]))                
                                    tokens_for_variables = list(tokens_for_variables)
                                    
                                    # For each token set
                                    for token_set in tokens_for_variables:
                                    
                                        # If tokens fit variables
                                        if rrs.matches_constraints(token_set, rule.var_dict_per_premise[j]):
                                                                    
                                            # Get variable dict information
                                            other_prem.var_dict = rule.var_dict_per_premise[j]
                                            token_var_dict = dict(zip(rule.var_dict_per_premise[j], token_set)) 
                                            final_var_dict = {**final_var_dict, **token_var_dict}
                                                                                                                                                            
                                            # Get all rule elements (seperately)
                                            elements = rule.condition_sets[j-1].elements   
        
                                            # Evalute elements to see if it can be successfully evaluated
                                            successfully_evaluated = rrs.evaluate_cond_elements(elements, other_prem, token_var_dict)
                   
                                            # If evaluated successfully with these settings
                                            # proceed to joint condition set
                                            if successfully_evaluated:
                                                
                                                # Get all rule elements (seperately)
                                                elements = rule.condition_sets[-1].elements  
                                                
                                                # Evalute elements to see if it can be successfully evaluated
                                                successfully_evaluated = rrs.evaluate_cond_elements(elements, premise, final_var_dict)
                        
                                                # If joint conditions match as well
                                                if successfully_evaluated:
                                                    
                                                    # Get elements
                                                    elements = rule.condition_sets[0].elements 
                                                    rule.condition.elements = elements
                                                
                                                    # Apply transitions and get new premises and nodes
                                                    new_premises, premise, root_node, finished = apply_transitions(rule, premise, TRANS, query, final_var_dict, new_premises, prem, root_node, full_tree=full_tree)
                                            
                                                    # If this rule should only be applied in limited fashion,
                                                    # save that premise/rule combination
                                                    if rule.rule_name in rule_settings.rules_with_limied_applition:  
                                                        # For each premise
                                                         for p in new_premises:
                                                             TRANS.limited_premises.append((rule.rule_name, p.premise_sent()))     
                                                      
                                                    # Stop processing if finished
                                                    if finished:
                                                        return True
                                                    
    # If all premises were processed, return False or new premises
    if not new_premises:
        return False
    else:
        return new_premises


def apply_transitions(rule, premise, TRANS, query, token_var_dict, 
                      new_premises, prem, root_node, verbose=False, full_tree=False):
    """Reconfirm evaluation and apply transitions."""
    
    # Get all rule elements (seperately)
    elements = rule.condition.elements   
    
    # Evalute elements to see if it can be successfully evaluated
    successfully_evaluated = rrs.evaluate_cond_elements(elements, premise, token_var_dict)
    
    # If rule matches
    if successfully_evaluated:   
                                       
        # If rule can be applied to same premise several times                                                              
        if rule.data in rule_settings.on_same_token_rules:
                                       
            limit = 3
            
            # Number of available hypernyms, hyponames etc. --> range or [None]
            n_hypernyms = range(premise.TRANS.available_hypernyms) if premise.TRANS.available_hypernyms > 0 else [None]
            n_hyponyms = range(premise.TRANS.available_hyponyms) if premise.TRANS.available_hyponyms > 0 else [None]
            n_synonyms = range(premise.TRANS.available_synonyms) if premise.TRANS.available_synonyms > 0 else [None]
            n_antonyms = range(premise.TRANS.available_antonyms) if premise.TRANS.available_antonyms > 0 else [None]
            n_siblings = range(premise.TRANS.available_siblings) if premise.TRANS.available_siblings > 0 else [None]
            
            # Combinations of all range lists
            combinations = list(itertools.product(*[n_hypernyms[:limit], n_hyponyms[:limit], n_synonyms[:limit], n_antonyms[:limit], n_siblings[:limit]]))
 
            # For each combo
            for comb in combinations:
                                                   
                # Set hypernym etc, number
                premise.TRANS.hypernym_nr, premise.TRANS.hyponym_nr, premise.TRANS.synonym_nr, premise.TRANS.antonym_nr, premise.TRANS.sibling_nr = comb
                
                for trans_set in rule.transitions.all_transition_sets:
                    
                    # Apply transition rule
                    transition_output = trans_set.apply(premise, token_var_dict, premise.TRANS.permutations_premise, verbose=verbose)   
                    
                    # If successful
                    if transition_output != False:
                        
                        # Get result from applied rule (best case: new premise)
                        new_premise, TRANS, tb_continued = check_new_premise_and_relation(rule, trans_set, premise, premise.TRANS, query, root_node, verbose=verbose)
   
                        # If full tree is to be evaluated
                        if full_tree:
                            if new_premise != None:
                                if not tb_continued:
                                    if not premise.words in TRANS.permutations_words and not premise.words in TRANS.all_branches:
                                 
                                        tree_node = Node(" ".join(new_premise.words), parent=root_node)    
#                                                            with open ('anytree_intermediate.json', 'w', encoding = 'utf-8') as f:
#                                                                exporter.write(root_node,f)                                                      
                                else:                                                     
                                    # Save new premise                                                    
#                                                        with open ('anytree_intermediate.json', 'w', encoding = 'utf-8') as f:
#                                                            exporter.write(root_node,f)     
#                                                        with open ('anytree_intermediate.json', 'w', encoding = 'utf-8') as f:
#                                                            exporter.write(root_node,f)
                                    with open('results/all_branches_and_rules.pkl','wb') as f: 
                                       pickle.dump((TRANS.all_branches, TRANS.all_applied_rules), f)    

                                    # Save new premise
                                    new_premises.append(new_premise)
                        
                        else:
                            # If new premise
                            if new_premise != None:
                                
                                if not tb_continued:
                                    if not premise.words in TRANS.permutations_words and not premise.words in TRANS.all_branches:
                                        tree_node = Node(" ".join(new_premise.words), parent=root_node)    
                                else:                                                     
                                    # Save new premise                                                    
                                    new_premises.append(new_premise)
                                    
                                    with open('results/all_branches_and_rules.pkl','wb') as f: 
                                       pickle.dump((TRANS.all_branches, TRANS.all_applied_rules), f)    
                                    
                                    # If match, done
                                    if TRANS.found_match:
                                        tree_node = Node(" ".join(new_premise.words), parent=root_node)    
                                        return new_premises, premise, root_node, True
                    
                    # Restore original premise
                    premise = copy.deepcopy(prem)
                    premise.TRANS = TRANS
                

        # If rule can be applied only once to each premise                                        
        else:     
                       
            # For each transition set
            for trans_set in rule.transitions.all_transition_sets:
                
                # Get transition output
                transition_output = trans_set.apply(premise, token_var_dict, TRANS.permutations_premise, verbose=verbose)
                
                # If transition outut is correct
                if transition_output != False:
                      
                    # Check the transition
                    new_premise, TRANS, tb_continued = check_new_premise_and_relation(rule, trans_set, premise, premise.TRANS, query, root_node, verbose=verbose)

                    # If full tree is to be computed
                    if full_tree:
                        if new_premise != None:
                                if not tb_continued:
                                    if not premise.words in TRANS.permutations_words and not premise.words in TRANS.all_branches:
                                        tree_node = Node(" ".join(new_premise.words), parent=root_node)    
#                                                            with open ('anytree_intermediate.json', 'w', encoding = 'utf-8') as f:
#                                                                exporter.write(root_node,f)                                                            
                                else:         
#                                                        with open ('anytree_intermediate.json', 'w', encoding = 'utf-8') as f:
#                                                            exporter.write(root_node,f)    
#                                                        # Save new premise                                                  
#                                                        with open ('anytree_intermediate.json', 'w', encoding = 'utf-8') as f:
#                                                            exporter.write(root_node,f)
                                    with open('results/all_branches_and_rules.pkl','wb') as f: 
                                       pickle.dump((TRANS.all_branches, TRANS.all_applied_rules), f)
                                    new_premises.append(new_premise)
                        
                    else:
                            # If new premise
                            if new_premise != None:
                                
                                # If no continued
                                if not tb_continued:
                                    
                                    # Add branch
                                    if not premise.words in TRANS.permutations_words and not premise.words in TRANS.all_branches:
                                        tree_node = Node(" ".join(new_premise.words), parent=root_node)    
                                else:                                                     
                                    # Save new premise                                                    
                                    new_premises.append(new_premise)
                                    
                                    # Save branches
                                    with open('results/all_branches_and_rules.pkl','wb') as f: 
                                       pickle.dump((TRANS.all_branches, TRANS.all_applied_rules), f)                                                            
                                    
                                    # If match, done
                                    if TRANS.found_match:
                                        tree_node = Node(" ".join(new_premise.words), parent=root_node)    
                                        return new_premises, premise, root_node, True

                # Restore original premise
                premise = copy.deepcopy(prem)                                  
                premise.TRANS = TRANS

    # Return new premises
    return new_premises, premise, root_node, False

def check_new_premise_and_relation(rule, trans_set, premise, TRANS, query, root_node, verbose=False):
    """Check resulting premise and check state, relation and other attributes."""
    
    # Get transitions input sentence as new premise
    premise = trans_set.transitioned_input_sent

    # Get projectivity
    projectivity = trans_set.projectivity 
    
    # Get projection, state and validity
    projection = rule.relation if projectivity == 'OPERATOR' else projectivity[rule.relation]
    new_state = rel_pol.transition_dict[(premise.state, projection)]
    validity_state = rel_pol.validity_state_dict[(premise.validity, new_state)]
                                 
    if verbose:
    
        print('\nRule applied:', rule.data)        
        print('\nCurrent state: ', premise.state)
        print('Transition relation: ', rule.relation)
        print('Projection: ', projection)
        print('New state: ', new_state)
        print('Validity:', validity_state)
     
    # If new premise is already in base
    if (premise.words) in TRANS.permutations_words:# and not (new_state == 'unknown' or validity_state == 'unknown') :
        
        if verbose:
            print('Already seen --> go back')
        
        # Quit
        return None, TRANS, False
#                                                                       
    # If unknown state or validity
    elif new_state.lower() == 'unknown' or validity_state.lower() == 'unknown':
        
        # Get state and validity
        premise.state = new_state 
        premise.validity = validity_state
        
        # Get new premise and save used rule
        new_premise = premise                                  
        new_premise.prev_rule = rule.condition.data
        
        # Save branches and rules
        TRANS.all_branches.append(" ".join([t.lower() for t in new_premise.words]))
        TRANS.all_applied_rules.append(rule.data)
        
        # Save 
        with open ('results/anytree_intermediate.json', 'w', encoding = 'utf-8') as f:
            exporter.write(root_node,f)
                               
        # If it matches query
        if query.match(premise):      
            
            # Fallback relation
            if premise.state != 'unknown':
                TRANS.fallback_relation = premise.state
            TRANS.fallback_validity = validity_state
            
            if verbose:
            
                print('Query:', query.string)
                print('Original premise:', TRANS.original_premise.string)
                print()
                print(TRANS.original_premise.string, rel_pol.rel_as_symbols[premise.state], query.string)
                print('')
                print('Relation:', premise.state)
                print('Inference validity:', premise.validity+'\n')
                
            else:
                TRANS.result = TRANS.original_premise.string+' '+rel_pol.rel_as_symbols[premise.state]+' '+query.string
                
            # Return new premise and stop
            return new_premise, TRANS, False
    
        # Unknown state
        else:
            
            if verbose:
                print('Unknown state --> go back')
        
            # Return new premise and stop
            return new_premise, TRANS, False
   
    # If not unknown state                    
    else:        
        
        # Get state, validity
        premise.state = new_state 
        premise.validity_state = validity_state
        
        # New premise information
        new_premise = premise
        new_premise.prev_rule = rule.condition.data
        
        # Save permutations, branches and rules
        TRANS.permutations_premise.append(new_premise)
        TRANS.permutations_words.append(new_premise.words)
        TRANS.all_branches.append(" ".join([t.lower() for t in new_premise.words]))
        TRANS.all_applied_rules.append(rule.data)
        
        # Save tree
        with open ('results/anytree_intermediate.json', 'w', encoding = 'utf-8') as f:
            exporter.write(root_node,f)
        
        # If it matches query
        if query.match(premise):       
            
            # Get final relation and validity
            TRANS.found_match = True
            TRANS.final_relation = premise.state
            TRANS.final_validity = validity_state
            
            if verbose:
            
                print('Query:', query.string)
                print('Original premise:', TRANS.original_premise.string)
                print()
                print(TRANS.original_premise.string, rel_pol.rel_as_symbols[premise.state], query.string)
                print('')
                print('Relation:', premise.state)
                print('Inference is', premise.validity_state+'\n')
                
            if not verbose:
                TRANS.result = TRANS.original_premise.string+' '+rel_pol.rel_as_symbols[premise.state]+' '+query.string
            
            # Return new premise, and finish
            return new_premise, TRANS, True
            
        else:
            # Return new premise, and finish
            return new_premise, TRANS, True
               

def main():
    """Testing"""
    
    # Set premise and query   
    premise = u"A is bigger than B"
    query = u"A is not greater than C"
    all_premises = [u"A is bigger than B",u"B is bigger than C"]

    premise = u"A Swede won a Nobel prize"
    query = u"A Scandinavian won a Nobel prize"
    all_premises = [u"A Swede won a Nobel prize", "Every Swede is a Scandinavian"]

    premise = u"If you eat vegetables, you stay healthy"
    query = u"You stay healthy"
    all_premises = [u"If you eat vegetables, you stay healthy", "You eat vegetables"]
    
    premise = u'Stimpy is a cat'
    query = u'Stimpy is not a poodle'
    all_premises = [premise]
    
    print('Hypothesis:')
    print(query)
    print('\nPremises:')
    for prem in all_premises:
        print('-', prem)
    print()

    # Parse query and premise    
    query_parse = nlp(query)
    premise_parse = nlp(premise)
    
    # Get tokens and sentence for query and premise
    query_tokens, query_sent = get_tokens_and_sent(query_parse)
    premise_tokens, premise_sent = get_tokens_and_sent(premise_parse)
    
    # Get all parsed premises
    all_parsed_premises = []

    for p in all_premises:
        if p == premise:
            continue
        parsed_p = nlp(p)
        p_tokens, p_sent = get_tokens_and_sent(parsed_p)
        p_Premise = Premise(p_tokens, p_sent)
        all_parsed_premises.append(p_Premise)
 
    # Save original premise and query
    original_premise = Premise(premise_tokens, premise_sent)
    premise = copy.deepcopy(original_premise)
    premise.other_premises = all_parsed_premises
    premise.original_premise = original_premise
    query = Query(query_tokens, query_sent)
    
    # Set polarity scope
    premise.set_polarity_scope()
    
    # Load rules
    rules = [rrs.Rule(r) for r in rule_settings.rule_set]
    mp_rules = [rrs.MP_Rule(r) for r in rule_settings.mp_rule_set]
    
    # Merge rules
    rules =  mp_rules + rules
    
    # Load rules
    for rule in rules:
        rule.load_rule(verbose=False)
        
    # Get root node and print tree
    root_node, TRANS = start_transformation_pipeline(rules, premise, query, verbose=True) 
    print('Inference Tree:\n')
    print_inference_tree(root_node)
       
    print('\nRelation:', TRANS.final_relation)
    print('Inference is', TRANS.final_validity)
    print()
    print(TRANS.result)
    
      
if __name__ == "__main__":
    main()
