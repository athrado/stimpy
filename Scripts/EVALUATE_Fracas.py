# -*- coding: utf-8 -*-
"""
@author: jsuter
Project: STIMPY - Sentence Transformative Infernece Mapping for Python

Julia Suter, 2019
---
EVALUATE_Fracas.py

- Load and preprocess FraCaS examples
- Select specific section of FraCaS
- Evaluate system on FraCaS examples
- Print performance overview
- Save intermediate results through logging
"""

# Import statements
import RUN_stimpy as stimpy
from RUN_stimpy import Premise, Query
from rule_reading_system import Rule, MP_Rule
import rule_settings
import wordnet_relations
import functions

import sys
import collections
import time
import pickle
import signal
import copy
import logging

logging.root.setLevel(logging.NOTSET)
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s %(message)s",
                    datefmt='%m-%d %H:%M',
                    filename='testing.log')

import spacy
import xml.etree.ElementTree as ET

from anytree import Node, RenderTree
from anytree.importer import JsonImporter

importer = JsonImporter()



# ----------------------------------------
### SETTINGS: Timeout, FraCaS subsets, WordNet
# ----------------------------------------

# Timoue after how many seconds
TIMEOUT = 15

FraCaS_IDs_total = ['001','002','003','004','005',
                    '006','007','008','009','010',
                    '011','015','017','022',
                    '023','024','025','029',
                    '030','031','032','033','038','039',
                    '040','041','045','046','047','048',
                    '054','055','056','057','058','059',
                    '060','063','064',
                    '070','071','072','073','074','075',
                    '076','079','080'
                    '197','198','199',
                    '201','202','203','204','205',
                    '206','207','216','217','218','219',
                    '221','225','229','230',
                    '231','233','234','237','239','240','249']   

FraCaS_IDs_quantifiers =  ['001','002','003','004','005',
                           '006','007','008','009','010',
                           '011','012','013','014','015',
                           '016','017','018','019','020',
                           '021','022','023','024','025',
                           '026','027','028','029',
                           '030','031','032','033','034',
                           '035','036','037','038','039',
                           '040','041','042','043','044',
                           '045','046','047','048','049',
                           '050','051','052','053','054',
                           '055','056','057','058','059',
                           '060','061','062','063','064',
                           '065','066','067','068','069',
                           '070','071','072','073','074',
                           '075','076','077','078','079','080']

FraCaS_IDs_adjectives = ['197','198','200',
                         '201','202','203','204','205','206',
                         '207','208','209','210',
                         '211','212','213','214',
                         '215','216','217','218','219']

FraCaS_IDs_comparatives = ['221','222','223','224','225',
                           '226','227','228','229','230',
                           '231','232','233','234','235',
                           '236','237','238','239','240',
                           '241','242','243','246','247','248','249']

# Pick one of the above FraCaS subsets, or None
fracas_selection = FraCaS_IDs_adjectives

# For None, select problem type
problem_type = ['single premise','multi premise','all'][-1]

# Include WordNet or not
use_wordnet = False

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

# Load Fracas samples/problems
tree = ET.parse('../Evaluation/FraCas/fracas.xml')
root = tree.getroot()
fracas_samples = root.findall('problem')

# Solution dict
sol_dict = {'yes':'valid',
            'no':'invalid',
            'unknown':'unknown',
            'undef':'undef'}

# ----------------------------------------
### FUNCTIONS: Printing, parsing and pipeline incl. Timeout function
# ---------------------------------------- 

class Timeout():
  """Timeout class using ALARM signal.
  Used for giving time constraint to problems."""
  class Timeout(Exception): pass

  def __init__(self, sec):
    self.sec = sec

  def __enter__(self):
    signal.signal(signal.SIGALRM, self.raise_timeout)
    signal.alarm(self.sec)

  def __exit__(self, *args):
    signal.alarm(0) # disable alarm

  def raise_timeout(self, *args):
    raise Timeout.Timeout()

def rule_frequencies(all_applied_rules, n=None):
    
    rule_counter=collections.Counter(all_applied_rules)
    if n is None:
        n = len(rule_counter)
    
    print()
    for rule, frequency in rule_counter.most_common(n):
        print(rule)
        print(frequency,'x')
        print()    

def parse_fracas_samples(child, verbose=False):
    """Parse FraCaS sample and segment into ID, premises, 
    hypothesis, and solution."""
    
    # Get premises
    premises = [p.text.strip() for p in child.findall('p')]
        
    # Get question and hypothesis
    q = child.find('q')
    h = child.find('h')
    
    question = q.text.strip()
    hypothesis = h.text.strip()
    
    # Get FraCaS answer and ID
    fracas_answer = child.attrib['fracas_answer']
    fracas_id =  child.attrib['id']
    
    # Print
    if verbose:
        print('*** EXAMPLE  '+str(fracas_id)+' ***')
        print('\nPremises:')
        for prem in premises:
            print('-', prem)    
        print()
    
        print('Hypothesis:')
        print(hypothesis)
        print()
        print('Can be inferred?', fracas_answer)
        print('\n****************************\n')
    
    # Return FraCaS sample elements
    return fracas_id, premises, hypothesis, sol_dict[fracas_answer]
      
    
def single_problem_evaluator(i,premise, query, query_tokens, 
                             all_parsed_premises, 
                             fallback_validity, fallback_relation, 
                             full_tree=False):
    """Process single premise-hypothesis problem (possibly multi-premise problem)
    and return computed validity and relation, and other information."""
            
    # Parse premise
    premise_parse = nlp(premise)
    premise_tokens, premise_sent = stimpy.get_tokens_and_sent(premise_parse)

    # Get other premises
    other_premises = [p for (j,p) in all_parsed_premises if j!=i]

    # Save original premise 
    original_premise = Premise(premise_tokens, premise_sent)
    premise = copy.deepcopy(original_premise)
    premise.original_premise = original_premise
    premise.other_premises = other_premises
    
    # Set polarity scope for premise
    premise.set_polarity_scope()
    

    # Wordnet settings
    if use_wordnet:
        
        # Get words to be looked up on Wordnet from premise
        wordnet_words_for_premise = [(t.lemma,t.u_pos) for t 
                                         in premise_tokens if t.u_pos 
                                         in ['NOUN','ADJ','ADV','VERB']]
    
        # Get words to be looked up on Wordnet from query
        wordnet_words_for_query = [(t.lemma,t.u_pos) for t in query_tokens 
                                       if t.u_pos 
                                       in ['NOUN','ADJ','ADV','VERB']]
        
        # Get Wordnet relations for these words
        wordnet_relations.get_all_wordnet_connections(wordnet_words_for_premise, 
                                                      wordnet_words_for_query)
        # Load Wordnet relations
        functions.load_wordnet_relations()   

    # Start inference pipeline
    root_node, PREMISE = stimpy.start_transformation_pipeline(rules, premise, query, 
                                                verbose=False, 
                                                full_tree=full_tree)
    # Print inference tree
    stimpy.print_inference_tree(root_node)

    print('\n**********************')
    print('Relation:', PREMISE.final_relation)
    print('Inference is', PREMISE.final_validity)

    # Save computed validity and relation
    computed_validity = PREMISE.final_validity
    computed_relation = PREMISE.final_relation
    
    # If validity not unknown, return results 
    if computed_relation not in ['UNKNOWN','unknown']:
        return (True, computed_validity, computed_relation, root_node, 
               fallback_validity, fallback_relation, PREMISE)
    
    # If unknown, check for fallback validity and return that
    else:
        try:
            computed_validity = PREMISE.fallback_validity
            computed_relation = PREMISE.fallback_relation
            
            fallback_validity = computed_validity
            fallback_relation = computed_relation
            
            return (False, computed_validity, computed_relation, root_node, 
                    fallback_validity, fallback_relation, PREMISE)
            
        except AttributeError:  
            return (False, computed_validity, computed_relation, root_node, 
                    fallback_validity, fallback_relation, PREMISE)
        
def evaluation_pipeline(fracas_samples, full_tree=False):
    """Pipeline for processing and testing single and multi premise problems."""
    
    # Get samples (discard "undef")
    samples =  [parse_fracas_samples(sample) for sample in fracas_samples]
    samples = [(fracas_id, premises, hypothesis, answer) 
                for (fracas_id, premises, hypothesis, answer) 
                in samples if answer != 'undef']
    
    # Initial settings
    v_correct = 0
    n_samples = 0    
    time_outs = 0
    
    correct_by_timeout = 0
    correct_by_default = 0
    correct_by_fallback = 0 
    
    correct_by_match = 0
    incorrect_by_match = 0
    
    correct_validities = []
    wrong_validities = []

    incorrectly_solved_samples = []
    correctly_solved_samples = []
    n_transitions = []
    all_applied_rules = [] 
    
    tp_unknown = 0
    tp_valid = 0
    tp_invalid = 0    

    fp_unknown = 0
    fp_invalid = 0
    fp_valid = 0
    
    incorrect_unknown = 0
    incorrect_valid = 0
    incorrect_invalid = 0
    
    total_valid = 0
    total_invalid = 0
    total_unknown = 0
    
    
    # If full tree is to be computed, prepare outfiles
    if full_tree:
        with open('trees.txt','w') as outfile:
            outfile.write('ALL TRANSITION TREES\n--------------------\n\n')
        with open('all_leaves.txt','w') as outfile:
            outfile.write('ALL TRANSITION TREES\n--------------------\n\n')
           
    # If there are no samples, quit
    if not samples:
        print('No samples')
        sys.exit()
    
    # For each sample...
    for samp_nr, sample in enumerate(samples):
        
        # Get sample information
        fracas_id, premises, hypothesis, validity = sample
        
        # Write out sample number
        if full_tree:
            with open('results/trees.txt','a') as outfile:
                outfile.write('\n---------------------------------------\n\n'
                              +str(samp_nr)+'\n')
                
            with open('results/all_leaves.txt','a') as outfile:
                outfile.write('\n---------------------------------------\n\n'
                              +str(samp_nr)+'\n')
        
        
        # If only subsection of FraCaS test is used, only use problems
        # for which IDs is given.  
        if fracas_selection is not None:
    
            if fracas_id not in fracas_selection:
                continue        
                
            # Filter out multi premise problems (except for selected ones)
            if len(premises)>1 and fracas_id not in ['002','003','004','011']:
                continue
            
        else:
            # Only process single premise problems
            if problem_type == 'single premise':
                
                # Filter out multi premise problems (except for selected ones)
                if len(premises)>1 and fracas_id not in ['002','003','004','011']:
                    continue         
 
            # Only process multi premise problems
            if problem_type == 'multi premise':                
                # Filter out single premise problems (plus selected multi problem ones)
                if not (len(premises) > 1 and fracas_id not in ['002','003','004','011']):
                    continue  
                
        print('\n*****************************\n')
        print('---------------')
        print(' Sample #', samp_nr+1)
        print('---------------')
        
        # Increment sample count
        n_samples += 1
        
        if validity == 'valid':
            total_valid += 1
        elif validity == 'invalid':
            total_invalid += 1
        else:
            total_unknown += 1
                
        # Parse query (hypothesis)
        query_parse = nlp(hypothesis)
        query_tokens, query_sent = stimpy.get_tokens_and_sent(query_parse)
        query = Query(query_tokens, query_sent)  
        
        print('Hypothesis:')
        print(hypothesis)
        print()
        
        # Settings
        fallback_validity = None
        fallback_relation = None
        
        timed_out = False
        is_finished = False
        
        all_parsed_premises = []
                    
        # Parse and save all premises for further processing
        for i,prem in enumerate(premises):
            
            # Parse and save all premises
            parsed_premise = nlp(prem)
            prem_tokens, prem_sent = stimpy.get_tokens_and_sent(parsed_premise)
            premise_instance = Premise(prem_tokens, prem_sent)
            all_parsed_premises.append((i,premise_instance))
        
        # For each premise
        for i, premise in enumerate(premises):
                        
            # Print
            print('Processing Premise', str(i+1),'...')
            print(premise)
            print() 
            
            # Save premise as string
            string_premise = premise
            
            # Track whether new rules are added 
            timeout_interruption = False
        
            # Run evaluator for problem with Timeout
            try: 
                with Timeout(TIMEOUT):
                    evaluator_output = single_problem_evaluator(i,premise, query, 
                                                       query_tokens, 
                                                       all_parsed_premises, 
                                                       fallback_validity, 
                                                       fallback_relation, 
                                                       full_tree=full_tree)
                    
                    (is_finished, computed_validity, computed_relation, 
                    root_node, fallback_vality, fallback_relation, 
                    PREMISE) = evaluator_output
                                       
            # If timeout is triggered...
            except Timeout.Timeout:     
    
                        # Label computed solution as timeout                   
                        computed_validity = 'timeout'
                        computed_relation = 'timeout'
                        timed_out = True
                        
                        # Load all created branches and used rules so far
                        with open('results/all_branches_and_rules.pkl', 'rb') as f:
                            loaded_results = pickle.load(f)
                            
                        # Add loaded number of transitions
                        n_transitions.append(len(set(loaded_results[0])))
                        
                        # Add loaded appplied rules
                        all_applied_rules += loaded_results[1]  
                        timeout_interruption = True
                    
                        # If creating the full tree...
                        if full_tree:
                            
                            # Load current root node
                            with open ('results/anytree_intermediate.json', 'r+', 
                                       encoding = 'utf-8') as f:
                                root_node = importer.read(f)
                                
                                # Write out processing information
                                with open('results/trees.txt','a') as outfile:
                                    outfile.write('Premise: '+string_premise+'\n')
                                    outfile.write('Hypothesis: '+hypothesis+'\n\n')                                
                                    outfile.write('Number transitions: '
                                                  +str(len(set(loaded_results[0])))+'\n')
                                    
                                    # Write out tree
                                    for pre, fill, node in RenderTree(root_node):
                                        out = "%s%s" % (pre, node.name)+'\n'
                                        outfile.write(out)
                                    outfile.write('\n\n')   
                                    
                                # Write out processing information
                                with open('results/all_leaves.txt','a') as outfile:
                                    outfile.write('Premise: '+string_premise+'\n')
                                    outfile.write('Hypothesis: '+hypothesis+'\n\n')
                                
                                    outfile.write('Number transitions: '
                                                  +str(len(set(loaded_results[0])))+'\n')
                                    
                                    # Write out branches
                                    for x in set(loaded_results[0]):
                                        outfile.write(x+'\n')
                                    outfile.write('\n')     
                                            
      
                    
            if not timeout_interruption:

                # Save transitions and applied rules
                print('# Total Transitions: '+str(len(set(PREMISE.all_branches))))                          
                n_transitions.append(len(set(PREMISE.all_branches)))                       
                all_applied_rules += PREMISE.all_applied_rules      
                               
                # If creating the full tree...
                if full_tree:
                    
                    # Add processing information
                    with open('results/trees.txt','a') as outfile:
                        outfile.write('Premise: '+string_premise+'\n')
                        outfile.write('Hypothesis: '+hypothesis+'\n\n')
                    
                        outfile.write('Number transitions: '
                                      +str(len(set(PREMISE.all_branches)))+'\n')
                        
                        # Write out tree
                        for pre, fill, node in RenderTree(root_node):
                            out = "%s%s" % (pre, node.name)+'\n'
                            outfile.write(out)
                        outfile.write('\n\n')   
                        
                    # Write out processing information
                    with open('results/all_leaves.txt','a') as outfile:
                        outfile.write('Premise: '+string_premise+'\n')
                        outfile.write('Hypothesis: '+hypothesis+'\n\n')
                    
                        outfile.write('Number transitions: '
                                      +str(len(set(PREMISE.all_branches)))+'\n')
                        
                        # Write out branches
                        for x in set(PREMISE.all_branches):
                            outfile.write(x+'\n')
                        outfile.write('\n')     
                            
     
            # If problem is solved, break
            if is_finished: 
                break
                   
            # If not "unknown" result, break
            if computed_relation not in ['UNKNOWN','unknown']:
                break
            
            # Otherwise, continue to search...
            
            # Write out tree information
            if full_tree:
                with open('results/trees.txt','a') as outfile:
                    outfile.write('Premise: '+string_premise+'\n')
                    outfile.write('Hypothesis: '+hypothesis+'\n\n')
                
                    for pre, fill, node in RenderTree(root_node):
                        out = "%s%s" % (pre, node.name)+'\n'
                        outfile.write(out)
                    outfile.write('\n\n')
                
        # Computed relation is "unknown"
        if computed_relation in ['UNKNOWN', 'unknown']:
            
            # Check for fallback valdity
            if fallback_validity != None:
                computed_validity = fallback_validity
                computed_relation = fallback_relation
                
                # Count if correct
                if validity == 'unknown':
                    correct_by_fallback += 1
              
        # Printing
        print('\n***** RESULTS *****')
        print('Correct answer:', validity)
        print('Computed answer:', computed_validity)
        print('Computed relation:', computed_relation)
        
        print('Time out:',timed_out)
        
        # EVALUATION:
        
        # If Timeout
        if computed_validity == 'timeout':
            if validity == 'unknown':
                correctly_solved = True
                correct_by_timeout += 1
                tp_unknown += 1
            else:
                correctly_solved = False
                fp_unknown += 1
                if validity == 'valid':

                    incorrect_valid +=1 
                if validity == 'invalid':
          
                    incorrect_invalid += 1
                    
        # If Default UNKNOWN
        elif computed_validity == 'UNKNOWN':
            if validity.lower() == 'unknown':
                print('Correct!')
                correctly_solved = True
                correct_by_default += 1
                tp_unknown += 1
            else:
                correctly_solved = False
                fp_unknown += 1
                print('Wrong...')
                
                if validity == 'valid':
                    incorrect_valid +=1 
                if validity == 'invalid':
                    incorrect_invalid += 1
         
        # If valid, invalid, unknown
        else:    
            if validity == computed_validity.lower():
                print('Correct!')
                correctly_solved = True
                if validity == 'valid':
                    tp_valid += 1

                if validity == 'invalid':
                    tp_invalid +=1 
                if validity == 'unknown':
                    tp_unknown +=1
                    
                correct_by_match +=1 
            else:
                print('Wrong...')
                correctly_solved = False
                incorrect_by_match += 1
                
                if validity == 'valid':
                    incorrect_valid += 1
                                      
                if validity == 'invalid':
                    incorrect_invalid += 1

                if validity == 'unknown':
                    incorrect_unknown += 1
                    
                if computed_validity.lower() == 'unknown':
                    fp_unknown += 1
                if computed_validity.lower() == 'valid':
                    fp_valid += 1                   
                if computed_validity.lower() == 'invalid':
                    fp_invalid += 1                    
                    
         
        # Save correctly and incorrectly solved sample IDs
        if correctly_solved:
            v_correct += 1
            correctly_solved_samples.append(fracas_id)
            correct_validities.append(validity.lower())
        else:
            incorrectly_solved_samples.append(fracas_id)
            wrong_validities.append(validity.lower())
            
        # Count timeouts
        if timed_out:
            time_outs += 1
            
            
    # Printing
    print('***********************************')
    print('   PERFORMANCE OVERVIEW   ')
    print('***********************************')
    print('Validity Accuracy:')
    print(str(v_correct)+'/'+str(n_samples))
    print(round(v_correct/n_samples*100.00,2))
    print('*********************')     
            
    # Logging
    logging.info('\n**************************')
    logging.info('PERFORMANCE OVERVIEW   \n**************************')
    logging.info('Validity Accuracy:')
    logging.info(str(v_correct)+'/'+str(n_samples))
    logging.info(str(round(v_correct/n_samples*100.00,2))+'%')
    logging.info('*********************')

    logging.info('Time outs: %d' % time_outs)
    logging.info('Guessed correctly by timeout: {}'.format(correct_by_timeout))
    logging.info('Correct by default: {}'.format(correct_by_default))
    logging.info('Correct by fallback: {}'.format(correct_by_fallback)+'\n')

    logging.info('Total valid: {}'.format(total_valid))
    logging.info('Total invalid: {}'.format(total_invalid))
    logging.info('Total unknown: {}'.format(total_unknown)+'\n')
    logging.info('Correct valid: {}'.format(tp_valid))
    logging.info('Correct invalid: {}'.format(tp_invalid))
    logging.info('Correct unknown: {}'.format(tp_unknown)+'\n')
    logging.info('Incorrect valid: {}'.format(incorrect_valid))
    logging.info('Incorrect invalid: {}'.format(incorrect_invalid))
    logging.info('Incorrect unknown: {}'.format(incorrect_unknown)+'\n')
    logging.info('FP valid: {}'.format(fp_valid))
    logging.info('FP invalid: {}'.format(fp_invalid))
    logging.info('FP unknown: {}'.format(fp_unknown)+'\n')

    logging.info('Correct by match: {}'.format(correct_by_match))
    logging.info('Incorrect by match: {}'.format(incorrect_by_match)+'\n')
    
    # Correct vs. incorrect
    correct_validities = collections.Counter(correct_validities)
    logging.info('Correct: {}'.format(correct_validities.most_common(3)))
        
    wrong_validities = collections.Counter(wrong_validities)
    logging.info('Wrong: {}\n'.format(wrong_validities.most_common(3)))
    
    logging.info('*********************')
    logging.info('Transitions')
    logging.info('Total: {}'.format(sum(n_transitions)))
    logging.info('Avg:   {}'.format(round(sum(n_transitions)/len(n_transitions)),2)) 
    logging.info('Min:   {}'.format(min(n_transitions)))
    logging.info('Max:   {}\n'.format(max(n_transitions)))
    
    # Rule frequencies
#    rule_frequencies(all_applied_rules, 5)
    
# ----------------------------------------
### RUN: call pipeline for evaluation
# ----------------------------------------

start = time.time()
evaluation_pipeline(fracas_samples)
end = time.time()

# Proessing time
logging.info('Processing time: {} minutes'.format(round((end - start)/60,2)))
