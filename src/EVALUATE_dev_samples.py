#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jsuter
Project: STIMPY - Sentence Transformative Infernece Mapping for Python

Julia Suter, 2019
---
EVALUATE_dev_samples.py

- Load development samples
- Evaluate samples
- Print performance results
"""

# Import statements
import copy
import sys
import time
import collections

import RUN_stimpy as stimpy
from RUN_stimpy import Premise, Query
from rule_reading_system import Rule, MP_Rule
import rule_settings
import wordnet_relations

import spacy
from anytree import Node, RenderTree
from anytree.exporter import DotExporter

# Settings
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------
### PREPROCESSING: Load spacy, rules and samples 
# ----------------------------------------

# Get spacy nlp for English
nlp = spacy.load('en')

# Get single-premise (sp) and multi-premise (mp) rules
sp_rules = [Rule(r) for r in rule_settings.rule_set]
mp_rules = [MP_Rule(r) for r in rule_settings.mp_rule_set]
rules =  sp_rules + mp_rules

# Load rules
for rule in rules:
    rule.load_rule(verbose=False)

# Load samples
with open("../Test Samples/test_samples.txt") as infile:
    samples_raw = infile.read()
    samples = samples_raw.split('\n\n')
    

# ----------------------------------------
### FUNCTIONS: Printing, parsing and pipeline
# ----------------------------------------

def print_rules():
    """Print all rules in use."""
    
    print('Rules:\n***********')
    for rule in rules:
        print('-  ',rule.data)
        
def rule_frequencies(all_applied_rules, n=5):
    
    rule_counter=collections.Counter(all_applied_rules)

    print()
    for rule, frequency in rule_counter.most_common(n):
        print(rule)
        print(frequency,'x')
        print()       
        
    
def parse_example(sample, verbose=False):
    """Parse samples into comment, premises, hypothesis, relation and validity."""

    # Split sample into lines
    lines = sample.split('\n')
    
    # Discard empty lines 
    lines = [line.strip() for line in lines if line.strip() != '']
    
    # Get sample comment
    sample_comment = lines[0]
    
    # Get premises (starting with -)
    premises = [line.split('-')[1].strip() for line in lines 
                if line.startswith('-')]

    # Get hypothesis
    hypothesis = lines[-3]
    hypothesis = hypothesis.split(':')[1].strip()
    
    # Get relation
    relation = lines[-2]
    relation = relation.split(':')[1].strip()
    
    # Get validity
    validity = lines[-1]
    validity = validity.split(':')[1].strip()
      
    # Print sample
    if verbose:
        
        print('\nExample', sample_comment)
        print('Premises:')
        for prem in premises:
            print('-', prem)
                
        print('Hypothesis:',hypothesis)
        print('Relation:',relation)
        print('Validity:',validity)
           
    # Return segmented and clean parts of sample
    return sample_comment, premises, hypothesis, relation, validity

    
    
def evaluation_pipeline(samples, full_tree=False):
    """Pipeline for processing and testing single/multi-premise samples."""
   
    # Initial settings
    correct_validity = 0
    correct_relation = 0    
    incorrectly_solved = []
    
    # Parse samples and filter out samples that are to be ignored
    samples = [parse_example(sample) for sample in samples]
    samples = [(sample_comment, premises, hypothesis, relation, validity) 
               for (sample_comment, premises, hypothesis, relation, validity) 
               in samples if 'ignore' not in sample_comment.lower()]
    
    # If full tree is created, prepare outfile
    if full_tree:
        with open('results/trees.txt','w') as outfile:
            outfile.write('ALL TRANSITION TREES\n--------------------\n\n')
    
    # Final number of samples
    n_examples = len(samples)
    
    # If there are no samples, exit 
    if not samples:
        print('No samples in this set.')
        sys.exit
    
    # Lists for collecting applied rules and transitions
    all_applied_rules = []    
    n_transitions = []
    
    # For each sample...
    for samp_nr, sample in enumerate(samples):
        
        print('---------------')
        print(' Sample #', samp_nr+1)
        print('---------------')
             
        # Get sample infos
        sample_comment, premises, hypothesis, relation, validity = sample
        
        # Parse query with spacy and save as Token and Sent instances
        query_parse = nlp(hypothesis)
        query_tokens, query_sent = stimpy.get_tokens_and_sent(query_parse)
        query = Query(query_tokens, query_sent)  
        
        # Validity and relation in case no better solution is found
        fallback_validity = None
        fallback_relation = None
        
        # List for premises
        all_parsed_premises = []
        
        # Parse and save all premises for further processing
        for i,prem in enumerate(premises):
            
            # Parse and save all premises
            parsed_premise = nlp(prem)
            prem_tokens, prem_sent = stimpy.get_tokens_and_sent(parsed_premise)
            premise_instance = Premise(prem_tokens, prem_sent)
            all_parsed_premises.append((i,premise_instance))
            
        # Print sample number
        if full_tree:
            with open('results/trees.txt','a') as outfile:
                outfile.write('\n---------------------------------------\n\n'
                              +str(samp_nr)+'\n')
    
        # For each premise
        for i, premise in enumerate(premises):
            
            # Print
            print('Processing Premise', str(i+1),'...')
            print(premise)
            print() 
            
            # Save premise as string
            string_premise = premise
            
            # Parse premise
            premise_parse = nlp(premise)
            premise_tokens, premise_sent = stimpy.get_tokens_and_sent(premise_parse)
            
            # Get the other premises
            other_premises = [p for (j,p) in all_parsed_premises if j!=i]
    
            # Save original premise 
            original_premise = Premise(premise_tokens, premise_sent)
            premise = copy.deepcopy(original_premise)
            
            # Save original and other premises as attributes to premise
            premise.original_premise = original_premise
            premise.other_premises = other_premises
            
            # Set polarity scop for premise
            premise.set_polarity_scope()
                        
            # Wordnet settings
            wordnet_sent_to_words_premise = [(t.lemma,t.u_pos) for t 
                                             in premise_tokens if t.u_pos 
                                             in ['NOUN','ADJ','ADV','VERB']]
            wordnet_relations.get_all_wordnet_connections(
                                                wordnet_sent_to_words_premise)
            
            # Process premise in inference pipeline
            root_node, PREMISE = stimpy.start_transformation_pipeline(rules, premise, query, 
                                                        verbose=False, 
                                                        full_tree=full_tree)
            
            # Print inference tree
            stimpy.print_inference_tree(root_node)
        
            # Save inference tree as picture
            DotExporter(root_node).to_picture("results/transformation_tree.png")
            
            # Save tree for each hypothesis-premise pair
            if full_tree:
                
                with open('results/trees.txt','a') as outfile:
                    outfile.write('Premise: '+string_premise+'\n')
                    outfile.write('Hypothesis: '+hypothesis+'\n\n')                
                    outfile.write('Number transitions: '
                                  +str(len(PREMISE.all_branches)))
                    
                    for pre, fill, node in RenderTree(root_node):
                        out = "%s%s" % (pre, node.name)+'\n'
                        outfile.write(out)
                    outfile.write('\n\n')
            
            # Print statements
            print('\n***** RESULTS *****')
    
            print('Relation:', PREMISE.final_relation)
            print('Inference is', PREMISE.final_validity)
            print()
            print('# Total transitions: '+
                  str(len(set(PREMISE.all_branches))))
            print()
                  
            # Save number of transitions for this hypothesis-premise pair
            n_transitions.append(len(set(PREMISE.all_branches)))
            # Save all applied rules
            all_applied_rules += PREMISE.all_applied_rules
    
            # Computed validity and relation
            computed_validity = PREMISE.final_validity
            computed_relation = PREMISE.final_relation
            
            # If computed relation not unknown, a solution was found
            if computed_relation not in ['UNKNOWN','unknown']:
                break
            # If computed relation is unknown or not found
            else:
                # If available, use fallback solution (usually 'unknown')
                try:
                    computed_validity = PREMISE.fallback_validity
                    computed_relation = PREMISE.fallback_relation
                    
                    fallback_validity = computed_validity
                    fallback_relation = computed_relation
                    break
                
                # Otherwise, assign "unknown"
                except AttributeError:
                    computed_validity = 'unknown'
                    computed_relation = 'unknown'
                    
        # If try using fallback solution if nothing else is found
        if computed_relation in ['UNKNOWN', 'unknown']:
            if fallback_validity != None:
                computed_validity = fallback_validity
                computed_relation = fallback_relation
            else:
                computed_validity = 'unknown'
                computed_relation = 'unknown'
                
            
        # Print solutions
        print('Correct answer: ', validity)
        print('Computed answer:', computed_validity)
        
        print('Correct relation: ', relation)
        print('Computed relation:', computed_relation)
        
            
        # Determine whether computed validity and relation are correct
        if validity == computed_validity:
            print('Correct!')
            correct_validity +=1
        else:
            print('Wrong...')
            # Save incorrect samples for later inspection
            incorrectly_solved.append((samp_nr, sample_comment))
             
        if relation == computed_relation:
            correct_relation += 1
            
        else:
            # Save incorrect samples for later inspection
            incorrectly_solved.append((samp_nr, sample_comment))
                
        print('\n********************************')
        
    # Print final results
        
    print('  PERFORMANCE OVERVIEW   ')
    print('********************************')
    print('Validity Accuracy:')
    print(str(correct_validity)+'/'+str(n_examples))
    print(round(correct_validity/n_examples*100.00,2))
    print()
    print('Relation Accuracy:')
    print(str(correct_relation)+'/'+str(n_examples))
    print(round(correct_relation/n_examples*100.00,2))
    print()
    print('Transitions')
    print('Total:', sum(n_transitions))
    print('Avg:  ', round(sum(n_transitions)/len(n_transitions),2))
    print('Min:  ', min(n_transitions))
    print('Max:  ', max(n_transitions))
    
    if get_most_frequent_rules:
        print('Most frequently used rules:\n')
        rule_frequencies(all_applied_rules)
        print()
    
    # Print incorrectly solved samples by number and comment
    if incorrectly_solved:
        print('\nProblematic samples:')
        for samp_nr, comment in set(incorrectly_solved):
            print(comment)

# ----------------------------------------
### RUN: call pipeline for evaluation
# ----------------------------------------

# Print most freuquently used rules
get_most_frequent_rules = False

start = time.time()
evaluation_pipeline(samples)
end = time.time()

# Proessing time
print('\nProcessing time:')
print(round((end - start)/60,2), 'minutes')

### RESULTS: 

# Problematic sampels:
# 19/20
# 33/34