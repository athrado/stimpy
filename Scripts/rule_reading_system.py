# -*- coding: utf-8 -*-
"""
@author: jsuter
Project: STIMPY - Sentence Transformative Infernece Mapping for Python

Julia Suter, 2019
---
rule_reading_system.py

- Load, interpret and preprocess rules
- Evaluate rules (condition set)
- Apply rules (transition set)
"""

# ----------------------------------------
### Import Statements
# ----------------------------------------

import rule_settings

import re
import regex
import copy
import itertools
import warnings

# ----------------------------------------
### HELPING FUNCTIONS for nested bracket handling
# ----------------------------------------

# Flatten list
flatten = lambda a : [element for sublist in [[b] if not isinstance(b,list) else b for b in a] for element in sublist]

def find_parens(s):
    """Get indices of matching parantheses in a string."""
    # Source: https://stackoverflow.com/questions/29991917/indices-of-matching-parentheses-in-python
    
    toret = {}
    pstack = []

    for i, c in enumerate(s):
        if c == '(':
            pstack.append(i)
        elif c == ')':
            if len(pstack) == 0:
                raise IndexError("No matching closing parens ')' at: " + str(i) + s)
            toret[pstack.pop()] = i

    if len(pstack) > 0:
        raise IndexError("No matching opening parens '(' at: " + str(pstack.pop())+s)

    return toret
    
def find_brackets(s):
    """Get indices of matching square brackets in a string."""
    # Source: https://stackoverflow.com/questions/29991917/indices-of-matching-parentheses-in-python
    
    toret = {}
    pstack = []

    for i, c in enumerate(s):
        if c == '[':
            pstack.append(i)
        elif c == ']':
            if len(pstack) == 0:
                raise IndexError("No matching closing parens ']' at: " + str(i))
            toret[pstack.pop()] = i

    if len(pstack) > 0:
        raise IndexError("No matching opening parens '[' at: " + str(pstack.pop()))

    return toret
    

def split_at_delimiter_outside_parentheses(string, delimiter_list):
        """Split string at given delimiter - but outside paranthesis or brackets."""
        
        # Get matching parantheses dicts
        parens_dict = find_parens(string)
        brackets_dict = find_brackets(string)
 
        # Get parentheses scope: all indices that are somehow embedded in parantheses
        parens_scopes = [range(key, parens_dict[key]) for key in parens_dict.keys()]        
        all_parens_scopes = [num for range_list in parens_scopes for num in range_list]
 
        # Get brackets scope: all indices that are somehow embedded in brackets       
        brackets_scopes = [range(key, brackets_dict[key]) for key in brackets_dict.keys()]        
        all_brackets_scopes = [num for range_list in brackets_scopes for num in range_list]

        # Find delimiters outside parantheses/bracket scopes
        delimiters = [(i,char) for (i,char) in enumerate(string) if char in delimiter_list and i not in all_parens_scopes and i not in all_brackets_scopes]     
        
        # Get indices of these delimiters (+ beginning of string)
        delimiter_chars = [char for (i,char) in delimiters]        
        indices = [i for (i,char) in delimiters]
        indices.insert(0,0)
        
        # Get elements of string (seperated by non-embedded delimiter)
        elements = [string[i:j].strip() if i==0 else string[i+1:j].strip() for i,j  in zip(indices, indices[1:]+[None])]
        
        # Return elements
        return elements, delimiter_chars

# ----------------------------------------
### CLASSES for lists
# ----------------------------------------


class OR_list():
    """List for managing OR elements in rule list"""
    
    def __init__(self,data, is_negation):        
        self.data = data
        self.is_negation = is_negation

    def __getitem__(self, item):
        return self.data[item]          
        
    def __str__(self):
        return "["+" | ".join([str(a) for a in self.data])+"]"
        
    def __len__(self):
        return len(self.data)
        
        
class AND_list():
    """List for managing AND elements in rule list"""
    
    def __init__(self,data, is_negation):        
        self.data = data
        self.is_negation = is_negation

    def __getitem__(self, item):
        return self.data[item]         
       
    def __str__(self):
        return "{"+" & ".join([str(a) for a in self.data])+"}"
        
    def __len__(self):
        return len(self.data)
        
# ----------------------------------------
### CLASSES for Predicates, Variables, Arguments
# ----------------------------------------
    
class Predicate():
    """Class for predicates."""
    
    def __init__(self, data, consider_constraints=True):        
        self.data = data
        self.constraints = []
        
        # Get predicate parts: actual predicate and brackets with arguments
        parts = re.search(r'(\w+)\((.*)\)', self.data)
        
        # Get actual predicate and arguments, e.g. hypernym + ['cat']
        self.pred = parts.group(1)
        argument_part = parts.group(2).strip()
        
        # Initialize lists
        self.arguments = []        
        self.variables = []
                
        # Get all arguments and delimiter characters (&, |)
        args, delimiter_chars = split_at_delimiter_outside_parentheses(argument_part, [','])
        
        # Clean all arguments and discard empty strings
        args = [arg.strip() for arg in args if arg != '']
        
        if consider_constraints:
            # Get POS constraints for arguments of this predicate
            self.constraints = rule_settings.pred_variable_limitations[self.pred]
            
            # If constraints do not match number of arguments, raise error
            if self.constraints != [] and len(args) != len(self.constraints):
                err_message = 'Incorrect number of arguments for predicate "'+self.pred+'"'
                raise IOError(err_message)

        # Save arguments and variables
        for i,arg in enumerate(args):
            
            # Get constraints
            constr = self.constraints[i] if self.constraints != [] else []
            arg = Argument(arg, constr)
            
            # Save
            self.arguments.append(arg)
            self.variables += arg.variables         
            
    def evaluate(self, input_sent, var_dict):
        """Evaluate predicate; compute wheter it returns True or False."""
    
        arguments = []
                
        # Iterate over every argument
        for arg in self.arguments:     
            
            # If arg is variable
            if arg.is_variable:     
                
                # If ends with _, it is a defined Token (such as NO, NOT)
                if arg.data.endswith('_'):
                    defined_token = copy.deepcopy(rule_settings.token_dict[arg.data])
                    arguments.append(defined_token)
                elif arg.data.startswith('!'):
                    defined_token = copy.deepcopy(rule_settings.token_dict['PLACEHOLDER_'])
                    arguments.append(defined_token)                    
                # If not, get Token through variable dict
                else:
                    arguments.append(var_dict[arg.data])
                    
            # If keyword argument
            elif arg.is_kwarg:

                # If variable
                if arg.kw_arg.is_variable:     
                        
                        # If ends with _, it is a defined Token (such as NO, NOT)
                        if arg.kw_arg.data.endswith('_'):
                            defined_token = copy.deepcopy(rule_settings.token_dict[arg.kw_arg.data])
                            arguments.append(defined_token)
                        elif arg.kw_arg.data.startswith('!'):
                            defined_token = copy.deepcopy(rule_settings.token_dict['PLACEHOLDER_'])
                            arguments.append(defined_token) 
                            
                        # If not, get Token through variable dict
                        else:
                            arguments.append(var_dict[arg.kw_arg.data])
                
                # If predicate
                elif arg.kw_arg.is_predicate:
                    arguments.append(Predicate(arg.kw_arg.data).evaluate(input_sent, var_dict))
                    
                # If condition
                elif  arg.kw_arg.is_condition:
                    arguments.append(Condition(arg.kw_arg.data).evaluate(input_sent, var_dict))
                    
                # If arg is string
                else:
                    arguments.append(arg)                    
                    
            # If arg is predicate, evaluate predicate and save
            elif arg.is_predicate:                                                                                     
                arguments.append(Predicate(arg.data).evaluate(input_sent, var_dict)) #                arguments.append(arg.evaluate(input_sent, var_dict)) # ORIGINALLY
                
            elif arg.is_condition:
                arguments.append(Condition(arg.data).evaluate(input_sent, var_dict))
                
            # If arg is string
            else:
                arguments.append(arg)

        # Call corresponding function with arguments
        return (rule_settings.predicate_dict[self.pred](arguments, input_sent))


class Variable():
    """Class for Variables."""
    
    def __init__(self,data, constraints):

        # Default settings
        self.data = data        
        self.string = None
        self.is_list = False
        self.has_constraints = False
       
        # Constraints
        self.constr_id = []
        self.constr_form = []
        self.constr_lemma = []
        self.constr_u_pos = constraints
        self.constr_x_pos = []
        self.constr_head = []
        self.constr_deprel = []
        
        # Negative constraints
        self.neg_constr_id = []
        self.neg_constr_form = []
        self.neg_constr_lemma = []
        self.neg_constr_u_pos = []
        self.neg_constr_x_pos = []
        self.neg_constr_head =  []
        self.neg_constr_deprel = []

        # If there is variable specification (designated by :)
        if ':' in self.data:    
            
            # Split at :
            split_variable = self.data.split(':')
            self.data = split_variable[0].strip()

            # Get constraints
            constr = split_variable[1].strip()
            constr = constr[1:-1]
            
            # If list of constants
            if not ('[' in constr or ';' in constr or '=' in constr):
                list_args = constr.split(',')
                self.is_list = True
                self.list_args = [arg.strip() for arg in list_args]
                self.n_args = len(list_args)
                
            # If there are constraints
            else:               
                constr = constr.split(';')
                self.has_constraints = True    
                
                # For each constraint
                for const in constr:
                    
                    # Get constraint type and value
                    parts = re.search('(.*?)\s*(!?=|(not)?\s*in)\s*(.*)',const)
                    constr_type = parts.group(1).strip()
                    operator = parts.group(2).strip()
                    value = parts.group(4).strip()
                    
                    # If "X (not) in [ ]"pattern
                    if 'in' in operator:
                    
                        # Split at comma and clean strings
                        constr_list = value[1:-1].split(',')
                        constr_list = [con.strip()[1:-1]  if con.startswith("'") and con.endswith("'") else con.strip() for con in constr_list]
                                                       
                        
                        # If "X not in [ ]" pattern
                        if 'not in' in operator:
                            setattr(self, 'neg_constr_'+constr_type, constr_list)
                            
                        # If "X in [ ]" pattern
                        else:
                            setattr(self, 'constr_'+constr_type, constr_list)
                            
                    # If "X (!)= something" pattern        
                    if '=' in operator:                
                        
                        # If unqueal
                        if '!=' in operator:
                            
                            # Check for conflicts with given pos constraints
                            if constr_type == 'u_pos' and (self.neg_constr_u_pos) and (value not in self.neg_constr_u_pos):
                                    raise IOError('POS constraints conflict')
                                    
                            setattr(self, 'neg_constr_'+constr_type, [value])
                            
                        # If equal               
                        else:
                            
                            # Check for conflicts with given pos constraints
                            if constr_type == 'u_pos' and (self.constr_u_pos) and (value not in self.constr_u_pos):
                                raise IOError('POS constraints conflict')
                                
                            setattr(self, 'constr_'+constr_type, [value])
                                                                
    def __str__(self):        
        return self.data

class Argument():
    """Class for Arguments (of Predicates and Functions)."""
    
    def __init__(self, data, constraints):
        
        self.data = data
           
        # Default settings
        self.constraints = constraints
        self.predicate = None 
        self.variables = []
        self.is_condition = False

        # Remove white space chars
        arg = data.strip()
                
        # Get predicate (if there is one)
        arg_predicate = re.search('\w+\(.*\)', arg)
        self.is_predicate = True if arg_predicate != None else False
                
        # Is argument digit? (+/- NUMBER)
        digits = re.search(r'^[\-\+]?\s*\d+$', self.data)
        self.is_digit = True if digits != None else False
        
        # Is argument string or variable or list?
        self.is_string = ((arg.startswith("'") and arg.endswith("'")) or arg.isdigit())
        self.is_kwarg = ('=' in arg and ':' not in arg) # ((not self.is_predicate) and 
        self.is_variable = ((not self.is_string) and (not self.is_predicate))  and (not self.is_digit) and (not self.is_kwarg)

        self.is_list = arg.startswith('[') and arg.endswith(']')
        
        # If argument is list of type [ ]
        if self.is_list:
            
            # List settings
            self.is_predicate = False
            self.is_digit = False
            self.is_string = False
            self.is_kwarg = False
            self.is_variable = False
            
            self.list_arguments = []
            
            # Get all ist arguments
            for list_arg in self.data[1:-1].split(','):
                list_arg =  Argument(list_arg.strip(), [])
                self.variables += list_arg.variables
                self.list_arguments.append(list_arg)
                    
        # If keyword argument
        if self.is_kwarg:
             
             # Get keyword and value
             self.is_predicate = False
             kwarg = re.search('(.*)=(.*)', self.data)
             self.keyword = kwarg.group(1).strip()
             self.kw_arg = kwarg.group(2).strip()

             # Add string ticks (') for string arguments missing those
             if self.keyword in ['left', 'right','deprel','lemma','form','id','head','x_pos','u_pos'] and self.kw_arg[0] != "'":
                 self.kw_arg = "'"+self.kw_arg+"'"
            
             # Make value a new Argument
             self.kw_arg = Argument(self.kw_arg, [])     
             
             # Get keyword argument variables
             self.variables += self.kw_arg.variables
                     
        # If string, remove surrounding quotes ('/")
        if self.is_string and not self.is_digit:
            self.data = arg[1:-1] if not arg.isdigit() else arg
            
        # If digit, get sign and value
        if self.is_digit:
            self.data = arg
            find_sign = re.search(r'[\-\+]', self.data)
            self.sign = find_sign.group(0) if find_sign != None else None
            self.value = int(re.search(r'\d+', self.data).group(0))
            
        # If predicate, evalute predicate and safe variables
        if self.is_predicate:
            
            # Get predicate and the rest
            predicate_string = arg_predicate.group(0)
            rest = arg[len(predicate_string):]

            # Equation
            if '=' in rest:
                self.is_predicate = False
                self.is_condition = True
                self.condition = Condition(arg)
                self.variables = self.condition.variables
                
            # No equation
            else:
                self.predicate = Predicate(predicate_string)
                self.variables += self.predicate.variables
            
        # If variable, save as Variable object
        if self.is_variable:
            var = Variable(self.data, self.constraints)           
            self.data = var.data             
            self.variables.append(var)
            
            
    def __str__(self):        
        return self.data

# ----------------------------------------
### CLASSES for Conditions, Transitions and Sets
# ----------------------------------------
        
class Condition():
    """Class for single Condition."""
    
    def __init__(self, data):
        self.data = data      
                
        # Match conditions
        matches = regex.search(r"\s*(((\w*(\(((?:[^\(\)]|(?R))*)\)))(\s*!?=\s*(\'?\w+\'?|\w*(\(((?:[^\(\)]|(?R))*)\)))?)?))\s*", str(self.data), flags=regex.R)       
    
        # Get first part (predicate with arguments)
        first_part = matches.group(3)
        
        # Get second part (if equation)
        sec_part = matches.group(6)
        
        # Is equation or not?
        self.is_equation = (sec_part != None)
        self.is_neg_equation = False
             
        self.variables =  []
        
        # If equation
        if self.is_equation:
            
            # If unequal
            if '!=' in sec_part:
                self.is_neg_equation = True
                self.parts = sec_part.split('!=')
         
            # If equal
            else:
                # Split at =
                self.parts = sec_part.split('=')
                
                
            # Second part as result (treated as Argument)
            self.result = Argument(self.parts[1].strip(), [])

            # Save variables
            if self.result:
                self.variables += self.result.variables
                
        # First part is predicate
        self.predicate = Predicate(first_part)        
        
        # Save variables for predicate
        self.variables += self.predicate.variables
        
        
    def evaluate(self, input_sent, var_dict):
        """Evaluate predicate; set in arguments and check if condition is fullfilled."""
        
        # Result of predicate
        pred_result = self.predicate.evaluate(input_sent, var_dict)

        # If equation
        if self.is_equation:

            # Get result or token through variable dict
            self.result = self.result if not self.result.is_variable else var_dict[self.result.data]
            
            # If Argument
            if isinstance(self.result, Argument):
                
                # If predicate, use result
                if self.result.is_predicate:
                    self.result = self.result.predicate.evaluate(input_sent, var_dict)
                self.result = str(self.result)

            # If negative equation, compare to result
            if self.is_neg_equation:
                return (pred_result != self.result)
            
            # If equation, compare to result
            else:          
                # Return whether predicate result matches result
                return (pred_result == self.result)
            
        else:
            return pred_result
         
    
class ConditionSet():
    """Class for set of conditions in rules"""
    
    def __init__(self, data):
        
        self.data = data.strip()
        self.var_dict = {} # variable dict
                
        # get all rule elements
        self.elements = seperate_conditions(self.data)
        
        
        # rule element as Condition objects
        self.cond_elements = [Condition(elem) for elem in extract_elements_from_nested_list(self.elements)]
        
        # Get all variables used in condition set
        var_list = [elem.variables for elem in self.cond_elements]  
    
        # Get variable list without duplicates
        self.variables = list(set([j for i in var_list for j in i]))
            
                        
class TransOperation():
    """Class for Transition Operation."""
        
    def __init__(self, data):        

        # Get data
        self.data = data
    
        # Get predicate parts: actual predicate and brackets with arguments
        parts = re.search(r'(\w+)\((.*)\)', self.data)

        # Get actual predicate and arguments, e.g. hypernym + ['cat']
        self.pred = parts.group(1)
        argument_part = parts.group(2)
        
        # Lists
        self.arguments = []        
        self.variables = []
        
        self.has_list_arg = False

        # Get all arguments and delimiters (&, |)       
        args, delimiter_chars = split_at_delimiter_outside_parentheses(argument_part, [','])
        
        # For each argument
        for i,arg in enumerate(args) :
                       
            # Save args as Argument object
            arg = Argument(arg, [])
            
            # Save in list
            self.arguments.append(arg)
            self.variables += arg.variables      
            
            if arg.is_list:
                self.has_list_arg = True
            

class TransitionSet():
    """Class for single Transition Set."""
    
    def __init__(self, data):
        
        # Get data and transition set
        self.data = data
        self.transition_set = [TransOperation(trans) for trans in self.data]
         
        # Get all variables
        var_list = [elem.variables for elem in self.transition_set]  
        self.variables = list(set([j for i in var_list for j in i]))
        
        # Get all variables
        arg_list = [elem.arguments for elem in self.transition_set]  
        self.arguments = list(set([j for i in arg_list for j in i]))
    
              
    def apply(self, input_sent, var_dict, permutations_input_sent, verbose=True):
        """Apply transition rules."""        
        
        # For each transition
        for transition in self.transition_set:
        
            arguments = []

            # Get arguments 
            for arg in transition.arguments:
                
                arg = copy.deepcopy(arg)
                
                # If variable, get it according to token dict or variable dict
                if arg.is_variable:                         
                    if arg.data.endswith('_'):
                        defined_token = copy.deepcopy(rule_settings.token_dict[arg.data])
                        arguments.append(defined_token)
                    else:
                        arguments.append(var_dict[arg.data])
                        
                # If keyword argument
                elif arg.is_kwarg:
                    
                    # If Argument
                    if isinstance(arg.kw_arg, Argument):
                        
                        # If arg is variable
                        if arg.kw_arg.is_variable:     
                                
                                # If ends with _, it is a defined Token (such as NO, NOT)
                                if arg.kw_arg.data.endswith('_'):
                                    defined_token = copy.deepcopy(rule_settings.token_dict[arg.kw_arg.data])
                                    arg.kw_arg = defined_token
                                    arguments.append(arg)
                                    
                                # If not, get Token through variable dict
                                else:
                                    result = var_dict[arg.kw_arg.data]
                                    arg.kw_arg = result
                                    arguments.append(arg)
                        
                        # If arg is predicate
                        elif arg.kw_arg.is_predicate:                                                                                    
                            result = Predicate(arg.kw_arg.data).evaluate(input_sent, var_dict) 
                            arg.kw_arg = result
                            arguments.append(arg)
                    
                        # If arg is condition
                        elif arg.kw_arg.is_condition:
                            result = Condition(arg.kw_arg.data).evaluate(input_sent, var_dict)
                            arg.kw_arg = result
                            arguments.append(arg)
                            
                        # If arg is string
                        else:
                            arguments.append(arg)
                            
                    else:                             
                        arguments.append(arg)
                    
                 # If predicate, evalute predicate
                elif arg.is_predicate:                    
                    result = Predicate(arg.data).evaluate(input_sent, var_dict)                
                    arguments.append(result)                   
                
                # If condition, get result for Condition
                elif arg.is_condition:
                    result = Condition(arg.data).evaluate(input_sent, var_dict)
                    arguments.append(result)
                else:
                    arguments.append(arg)
                
            # If one of the arguments could not be determine, do not apply transitions
            if None in arguments:
                warnings.warn('Arguments failed')
                input_sent = permutations_input_sent[-1]
                return False
                
            # Get input_sent after transition rules were applied, get projectivity dict
            self.transitioned_input_sent, self.projectivity = rule_settings.trans_operation_dict[transition.pred](arguments, input_sent, verbose=verbose)  
                        
            # Check for failed projectivity computation
            if self.projectivity == None or self.transitioned_input_sent == None:
                warnings.warn('Projectivtiy failed')
                return False
                                  
        # Return True to indicate successfull transition
        return True
            

       
class TransitionSets():
    """Class for complete Transition Sets."""

    def __init__(self, data):
        
        # Get data and seperate into elements
        self.data = data         
        self.elements = seperate_conditions(self.data)
        
        # Expand transitions
        all_transition_options = expand(self.elements)
             
#        print('***** TRANSITION OPTIONS *******')
#        for x in all_transition_options:
#            print(x)
#        print('*********')
       
        # Get all tranisition sets (after expansion)
        self.all_transition_sets = [TransitionSet(option) for option in all_transition_options]

        # Get arguments
        self.arguments = [elem.arguments for elem in self.all_transition_sets]  
        self.arguments = list(set([j for i in self.arguments for j in i]))
        
        # Get variables
        self.variables = [elem.variables for elem in self.all_transition_sets]  
        self.variables = list(set([j for i in self.variables for j in i]))   
            


# ----------------------------------------
### CLASSES for Rules
# ----------------------------------------
   

class Rule():
    """Class for Single-Premise Rules."""
    
    def __init__(self, data): 
        
        # Default settings
        self.data = data
        self.original_data = self.data
        
        self.multipremise_rule = False        
        self.rule_name =  None
        
        # Get rule name
        if self.data.startswith('~'):              
    
            find_parts = re.search(r'~(.*)\n(.*)', self.data)
            self.rule_name = find_parts.group(1).strip()
            self.data = find_parts.group(2).strip()
            
            if self.rule_name.endswith('~'):
                self.rule_name = self.rule_name[:-1].strip()
                    
        
    
    def create_var_dict(self):
        """Create a variable dict for variables and matching tokens."""
        
        self.var_dict = {}
        
        # Get all variables
        all_variables = list(set(self.condition.variables + self.transitions.variables))
        
        # Get constant variables (ending with _) and normal ones
        self.const_variables = [var for var in all_variables if var.data.endswith('_')  or var.is_list]
        self.variables = [var for var in all_variables if not var.data.endswith('_') and not var.is_list]
        self.const_variables_list = {var.data : var.list_args for var in all_variables if not var.data.endswith('_') and var.is_list}
   
        # All constraint attributes
        constraint_attributes = ['lemma','form','x_pos','u_pos','deprel','id']        
        
        # For each variable
        for var in self.variables:
                                        
            # If variable unseen, add to dict
            if var.data not in self.var_dict.keys():                    
                self.var_dict[var.data] = var
        
            # If variable seen already
            else:
                
                # Create placeholder
                placeholder_var = self.var_dict[var.data]
                
                # For each constraint 
                for constr_type in constraint_attributes:
                                        
                    # Get current and new constraint
                    constr = getattr(placeholder_var, 'constr_'+constr_type)
                    new_constr = getattr(var, 'constr_'+constr_type)
                    
                    # Get current and new negative constraint
                    neg_constr = getattr(placeholder_var, 'neg_constr_'+constr_type)
                    new_neg_constr = getattr(var, 'neg_constr_'+constr_type)
                    
                    # If only new constraint, use that
                    if new_constr and not constr:
                        setattr(placeholder_var, 'constr_'+constr_type, new_constr)                       
                        
                    # If both current and new constraint, use intersection
                    if new_constr and constr:
                        intersection = list(set(constr) & set(new_constr)) 
                        setattr(placeholder_var, 'constr_'+constr_type, intersection)
                        
                    # If only new constraint, use that
                    if new_neg_constr and not neg_constr:
                        setattr(placeholder_var, 'neg_constr_'+constr_type, new_neg_constr)                       
                        
                    # If both current and new negative constraint, use union
                    if new_neg_constr and neg_constr:
                         union = list(set(neg_constr) | set(new_neg_constr))
                         setattr(placeholder_var, 'neg_constr_'+constr_type, union)
                         
                     # Save new constraint data
                    self.var_dict[var.data] = placeholder_var
                      
        # Get variables
        self.variables = [var for var in self.variables if not var.data.startswith('!')]
        
        # Get full variable dict (with negative variables !)
        self.full_var_dict = self.var_dict        
        
        # Get var dict (without negative variables)
        self.var_dict = {k:self.full_var_dict[k] for k in self.full_var_dict if not k.startswith('!')}
        
        # Get var dict (with only negative variables)
        self.exclusive_var_dict = {k:self.full_var_dict[k] for k in self.full_var_dict if k.startswith('!')}
                
        # Number of variables (not constants)                
        self.n_var = len(self.var_dict.keys())        

    def load_rule(self, verbose=False, unit_test=False):
        """Load all rules."""
            
        # Split at '-->' into condition and transitions
        rule_split = self.data.split('-->')
        
        self.second_part = rule_split[1].split('#')

        # Get condition and tranisition sets
        self.condition = ConditionSet(rule_split[0])             
        self.transitions =  TransitionSets(self.second_part[0].strip())

        # Metadata 
        self.meta_data = self.second_part[1].strip()
        
        # Relation
        self.relation = (re.search(r'relation\s*=(.*)', self.meta_data).group(1)).strip()

        # Create variable dct
        self.create_var_dict()
        
        if verbose:        
            # For each condition
            print('\nElements:')
            for cond in self.condition.cond_elements:
                
                # Print Condition, Predicate, Arguments
                print('')
                print(cond.data,'\n')
                print('\tPredicate:', cond.predicate.data)
                print('\tArguments:', [arg.data for arg in cond.predicate.arguments])
                
                # For each argument
                for arg in cond.predicate.arguments:
                    
                    # Print Predicate and Arguments for predicate arguments
                    if arg.is_predicate:
                        print('\t\tPredicate:', arg.predicate.data)
                        print('\t\tArguments:', [arg.data for arg in arg.predicate.arguments])
                
                # Print Predicate and Arguments for result in equation
                if cond.is_equation:
                    print('\tEquation result:',cond.result.data)
                    
                    if cond.result.predicate != None:
                        print('\t\tPredicate:', cond.result.predicate.data)
                        print('\t\tArguments:', [arg.data for arg in cond.result.predicate.arguments])
            
            print('\n************\n')        
            
            # Print conditions
            for cond in self.condition.elements:
                print(cond)
                
        # For unit test (incomplete)
        if unit_test:
                         
            results = []
            for cond in self.condition.cond_elements:
                               
                cond_result = []
                cond_result.append(cond.data+'\n')
                cond_result.append('\tPredicate: '+cond.predicate.data)
                cond_result.append('\tArguments: ['+ ",".join([arg.data for arg in cond.predicate.arguments])+']')
                 
                for arg in cond.predicate.arguments:
                    
                    if arg.is_predicate:
                        cond_result.append('\t\tPredicate: '+arg.predicate.data)
                        cond_result.append('\t\tArguments: ['+ ",".join([arg.data for arg in arg.predicate.arguments])+']')
                     
                # Print Predicate and Arguments for result in equation
                if cond.is_equation:
                    cond_result.append('\tEquation result:'+cond.result.data)
                     
                    if cond.result.predicate != None:
                        cond_result.append('\t\tPredicate: '+ cond.result.predicate.data)
                        cond_result.append('\t\tArguments: ['+",".join([arg.data for arg in cond.result.predicate.arguments])+']')
                            
                results.append('\n'.join(cond_result))
                
            return '\n'.join(results)                
   

 
class MP_Rule():
    """Class for Multi-Premise Rules."""
    
    def __init__(self, data):
        
        # Default settings
        self.data = data
        self.original_data = self.data        
        self.multipremise_rule = True        
        self.rule_name = None
        
        # If rule name is introduced      
        if self.data.startswith('~'):              
            
            # Extract rule name
            data_splits = self.data.split('\n')
            self.rule_name = data_splits[0][1:].strip()
            self.rule_name = self.rule_name[:-1] if self.rule_name.endswith('~') else self.rule_name
            self.data = ' '.join(data_splits[1:])
        
    
    def create_var_dict(self):
        """Create a variable dict for variables and matching tokens."""
        
        self.var_dict = {}
        
        # Get all variables
        all_variables = list(set(self.variables))
        
        # Get constant variables (ending with _) and normal ones
        self.const_variables = [var for var in all_variables if var.data.endswith('_')  or var.is_list]
        self.variables = [var for var in all_variables if not var.data.endswith('_') and not var.is_list]
        self.const_variables_list = {var.data : var.list_args for var in all_variables if not var.data.endswith('_') and var.is_list}
    
        # All constraint attributes
        constraint_attributes = ['lemma','form','x_pos','u_pos','deprel','id']        
        
        # For each variable
        for var in self.variables:                            
            
            # If variable unseen, add to dict
            if var.data not in self.var_dict.keys():                  
                self.var_dict[var.data] = var
        
            # If variable seen already
            else:
                
                # Create placeholder
                placeholder_var = self.var_dict[var.data]
                
                # For each constraint attribute
                for constr_type in constraint_attributes:                    
                    
                    # Get current and new constraint
                    constr = getattr(placeholder_var, 'constr_'+constr_type)
                    new_constr = getattr(var, 'constr_'+constr_type)
                    
                    # Get current and new negative constraint
                    neg_constr = getattr(placeholder_var, 'neg_constr_'+constr_type)
                    new_neg_constr = getattr(var, 'neg_constr_'+constr_type)
                    
                    # If only new constraint, use that
                    if new_constr and not constr:
                        setattr(placeholder_var, 'constr_'+constr_type, new_constr)                       
                        
                    # If both current and new constraint, use intersection
                    if new_constr and constr:
                        intersection = list(set(constr) & set(new_constr)) 
                        setattr(placeholder_var, 'constr_'+constr_type, intersection)
                        
                    # If only new negative constraint, use that
                    if new_neg_constr and not neg_constr:
                        setattr(placeholder_var, 'neg_constr_'+constr_type, new_neg_constr)                       
                        
                    # If both current and new constraint, use union of them
                    if new_neg_constr and neg_constr:
                         union = list(set(neg_constr) | set(new_neg_constr))
                         setattr(placeholder_var, 'neg_constr_'+constr_type, union)
                         
                    # Save new constraint data
                    self.var_dict[var.data] = placeholder_var
                      
        
        # Get variables
        self.variables = [var for var in self.variables if not var.data.startswith('!')]
        
        # Get full variable dict (with negative variables !)
        self.full_var_dict = self.var_dict        
        
        # Get var dict (without negative variables)
        self.var_dict = {k:self.full_var_dict[k] for k in self.full_var_dict if not k.startswith('!')}
        
        # Get var dict (with only negative variables)
        self.exclusive_var_dict = {k:self.full_var_dict[k] for k in self.full_var_dict if k.startswith('!')}
                
        # Number of variables (not constants)                
        self.n_var = len(self.var_dict.keys())
                
                
    def load_rule(self, verbose=True, unit_test=False):
        """Load all rules."""
          
        # Split at '-->' into condition and transitions
        rule_split = self.data.split('-->')
    
        # Split second part from metadata
        self.second_part = rule_split[1].split('#')
                                   
        # Split different condition sets
        condition_sets = rule_split[0].split('---')
        self.condition_sets = []
        self.variables = []
    
        # Start variable dicts
        self.var_in_premise_condset = {}
        self.var_dict_per_premise = {}
        
        # Get number of condition sets
        nr_condition_sets = len(condition_sets)
        
        # For each conditions et
        for i,cond_set in enumerate(condition_sets):
            
            # Clean
            cond_set = cond_set.strip()
            
            # Make into ConditionSet instance
            if cond_set.startswith('\t'):
                cond_set = re.search(r'\t(.*)',cond_set).group(1).strip()
            cond_set = ConditionSet(cond_set)

            # Save
            self.condition_sets.append(cond_set)
            self.variables += cond_set.variables
            
            # Don't process last part
            if i == nr_condition_sets-1:
                continue
                                 
            # Get condition set variable dict
            self.var_in_premise_condset[i+1]  = []
            
            for var in cond_set.variables: 
                    self.var_in_premise_condset[i+1].append(var)
                    
        # Get condition
        self.condition = ConditionSet(rule_split[0])         
        
        # Get transition sets
        transition_sets = self.second_part[0].strip().split('---')
        self.transition_sets = []
        
        # For each transition set
        for trans_set in transition_sets:
            
            # Clean 
            trans_set = trans_set.strip()
            trans_set = re.search(r'.*?:(.*)',trans_set).group(1).strip()
            trans_set = TransitionSets(trans_set)
            
            # Save
            self.transition_sets.append(trans_set)
            self.variables += trans_set.variables
        
            # Process variables
            for var in trans_set.variables:
                if var.has_constraints and var not in self.var_in_premise_condset[i+1]:
                    self.var_in_premise_condset[i+1].append(var)
            
        # Get transiitons
        self.transitions = self.transition_sets[0]

        # Metadata 
        self.meta_data = self.second_part[1].strip()
        
        # Relation
        self.relation = (re.search(r'relation\s*=(.*)', self.meta_data).group(1)).strip()

        # Create variable dict
        self.create_var_dict()
        
        # for each key for variables in the premise 
        for key in self.var_in_premise_condset:
            new_dict = {}
            for var in self.var_in_premise_condset[key]:
                new_dict[var.data]  = self.var_dict[var.data]
            self.var_dict_per_premise[key] = new_dict

        # Printing
        if verbose:        
            # For each condition
            print('\nElements:')
            for cond in self.condition.cond_elements:
                
                # Print Condition, Predicate, Arguments
                print('')
                print(cond.data,'\n')
                print('\tPredicate:', cond.predicate.data)
                print('\tArguments:', [arg.data for arg in cond.predicate.arguments])
                
                # For each argument
                for arg in cond.predicate.arguments:
                    
                    # Print Predicate and Arguments for predicate arguments
                    if arg.is_predicate:
                        print('\t\tPredicate:', arg.predicate.data)
                        print('\t\tArguments:', [arg.data for arg in arg.predicate.arguments])
                
                # Print Predicate and Arguments for result in equation
                if cond.is_equation:
                    print('\tEquation result:',cond.result.data)
                    
                    if cond.result.predicate != None:
                        print('\t\tPredicate:', cond.result.predicate.data)
                        print('\t\tArguments:', [arg.data for arg in cond.result.predicate.arguments])
            
                
            print('\n************\n')        
            
            # Print conditions
            for cond in self.condition.elements:
                print(cond)
                
        # For unit testing... (incomplete)
        if unit_test:
            
            results = []
            for cond in self.condition.cond_elements:
                               
                cond_result = []
                cond_result.append(cond.data+'\n')
                cond_result.append('\tPredicate: '+cond.predicate.data)
                cond_result.append('\tArguments: ['+ ",".join([arg.data for arg in cond.predicate.arguments])+']')
                 
                for arg in cond.predicate.arguments:
                    
                    if arg.is_predicate:
                        cond_result.append('\t\tPredicate: '+arg.predicate.data)
                        cond_result.append('\t\tArguments: ['+ ",".join([arg.data for arg in arg.predicate.arguments])+']')
                     
                     
                # Print Predicate and Arguments for result in equation
                if cond.is_equation:
                    cond_result.append('\tEquation result:'+cond.result.data)
                     
                    if cond.result.predicate != None:
                        cond_result.append('\t\tPredicate: '+ cond.result.predicate.data)
                        cond_result.append('\t\tArguments: ['+",".join([arg.data for arg in cond.result.predicate.arguments])+']')
                                        
                results.append('\n'.join(cond_result))
                
            return '\n'.join(results)   
        
        
 

# ----------------------------------------
### FUNCTIONS for handling expanding list arguments
# ----------------------------------------
  
def expand_list_argment(transition):
    """Expand list arguments for transitions and return as OR list:
    For example: A & [B_, C_] --> (A & B_) | (A & C_)."""
    
    # Get predicate and arguments
    predicate = transition.pred
    arguments = [arg.list_arguments if arg.is_list else [arg] for arg in transition.arguments]
    
    # Expansion
    expanded_transitions = []
    
    # Combine arguments and expand
    for comb in itertools.product(*arguments):
    
        # Get new transitions
        new_transition = predicate + '(' + ', '.join([c.data for c in comb]) + ')'
        expanded_transitions.append(new_transition)
        
    # Return as OR list
    return OR_list(expanded_transitions, False)


def expand(elements):
    """Expand Transition Operations."""
    
    transitions = []

    # If string
    if isinstance(elements, str):

        trans_x = TransOperation(elements)
        
        # Expand if there is list argument
        if trans_x.has_list_arg:
            expanded_trans_list = expand_list_argment(trans_x)
            intermediate = expand(expanded_trans_list)                    
            transitions += flatten(intermediate)
            
        else:
            transitions.append(elements)
            
        return transitions
    
    # If OR list
    if isinstance(elements, OR_list):
        
        # For each element
        for elem in elements:

            # If string
            if isinstance(elem, str):
                
                trans_x = TransOperation(elem)
                
                # Expand if there is list argument
                if trans_x.has_list_arg:
                    expanded_trans_list = expand_list_argment(trans_x)
                    intermediate = expand(expanded_trans_list)                    
                    transitions += flatten(intermediate)
                        
                else:
                    
                    transitions.append(elem)      
                
            # If AND list
            if isinstance(elem, AND_list):
                
                # Expand AND list
                intermediate = expand(elem)
                transitions += intermediate
                
            # If OR list
            if isinstance(elem, OR_list):
     
                # Expand OR list
                intermediate = expand(elem)
                transitions += flatten(intermediate)
                
        # Get all element combos
        all_element_combos = [list(elem) for elem in itertools.product(*[transitions])]
        # Flatten (no embedded structures)
        all_element_combos = [flatten(u) for u in all_element_combos]
        
        return all_element_combos
    
    # If AND list
    if isinstance(elements, AND_list):
        
        transitions = []
        
        # Expand every element
        for elem in elements:
            intermediate = expand(elem)
            transitions.append(intermediate)
            
        # Get all element combos
        all_element_combos = [list(elem) for elem in itertools.product(*transitions)]
        # Flatten (no embedded structures)
        all_element_combos = [flatten(u) for u in all_element_combos]
            
        return all_element_combos

def get_elements(string, verbose=False):
    """Get separate elements as OR list."""
    
    # Settings
    is_negation = False 
    
    if verbose:
        print('String:')
        print(string)
        
    # Remove "not" from predicate, set to is_negation
    if string.startswith('not('):
        string = string[4:-1]
        is_negation = True
    
    # Count brackets
    opening_brackets = string.count('(')
    closing_brackets = string.count(')')
    
    # Raise error if unequal number of brackets
    if opening_brackets != closing_brackets:
        raise IOError('Unequal number of brackets in rule:\n'+string)
        
    # Get condition parts and separating log operators
    condition_parts, log_operators = split_at_delimiter_outside_parentheses(string, ['&','|'])
    
    # Get pairs of condition parts: (a,b),(b,c),(c,d)
    zipped_conditions = list(zip(condition_parts[0:], condition_parts[1:]))
    
    # If there is logical operator
    if len(log_operators)>0:
        
        # Remove unnecessary brackets                            
        zipped_conditions = [(part_1, part_2)  for (part_1, part_2) in zipped_conditions]
      
        if verbose:
            print('Functions:')
            print(log_operators)    
    
        # Settings
        outer_level_conds = []      
        curr_group = []
        
        # For each pair                
        for i, cond_pair in enumerate(zipped_conditions):            
            
            # At the beginning simply take first element
            if i == 0:                
                curr_group.append(cond_pair[0])
            
            # If & operator, append to AND list
            if log_operators[i] == '&':                
                curr_group.append(cond_pair[1])
                
            # If | operator, append current group of elements to OR list
            # Either whole list or single element
            else:
                if len(curr_group)>1:
                    outer_level_conds.append(AND_list(((curr_group)), is_negation))
                else:
                    outer_level_conds.append(curr_group[0])
                   
                # Add next element to current group
                curr_group = [cond_pair[1]]                
            
            # If last element    
            if i == len(zipped_conditions)-1:
                
                # Append current group of elements to OR list
                if len(curr_group)>1:
                    outer_level_conds.append( AND_list(((curr_group)),is_negation) )
                else:
                    outer_level_conds.append(curr_group[0])              
    else:
        # Remove unnecessary brackets
        condition = condition_parts[0]
        condition = condition[1:-1] if condition.startswith('(') and condition.endswith(')') else condition
        
        # Return as OR list
        return OR_list([condition],is_negation)
       
    if verbose:
        print('Conditions:')
        for cond in outer_level_conds:
            print(cond)
                    
    # Return outer_level_conds
    return OR_list(outer_level_conds,is_negation)

# ----------------------------------------
### FUNCTIONS for handling conditions
# ----------------------------------------        

def seperate_conditions(string):  
    """Sperate conditions from given string."""
    
    # Settings
    elements = []
    is_negation = False 
            
    # Remove "not" from predicate, set to is_negation
    if string.startswith('not('):
        string = string[4:-1]
        is_negation = True
        
    # Get outmost level conditions
    outer_level_conds = get_elements(string)    
    
    # For each group
    for cond in outer_level_conds:

        # If not AND list
        if not isinstance(cond, AND_list):

                # If more logical operators in expression, split and save
                if '&' in cond or '|' in cond: 
                    parts = seperate_conditions(cond)
                    elements.append(parts)
                else:
                    elements.append(cond)

        # IF AND list
        else:
                all_parts = []     
                
                # If more logical operators in expression, split and save
                for part in cond:
                    if '&' in part or '|' in part:
                        parts = seperate_conditions(part)
                        
                        # If only one part, save only first element (no unnecessary brackets)
                        if len(parts)==1:
                            all_parts.append(parts[0])
                        else:
                            all_parts.append(parts)                
                    else:
                        all_parts.append(part)
                        
                # Append as AND list
                elements.append(AND_list(all_parts, is_negation))

    # IF AND list and only 1 element, return only first element (no unnecessary brackets)
    if len(elements)==1 and isinstance(elements[0], AND_list):
        final_elements = elements[0]
        
    else:
        final_elements =  elements
        
    # If AND, return as AND list
    if isinstance(final_elements, AND_list):
        return AND_list(final_elements, is_negation)

    # Otherwise return as OR list
    else:
        return OR_list(final_elements, is_negation)           


def reverse_boolean(boolean):
    """Invert boolean."""
    
    if boolean == True:
        return False
    else:
        return True

def evaluate_cond_elements(elem, input_sent, var_dict):
    """Evaluate single condition elements. Stop evaluation as soon as possible."""
    
    # If AND list
    if isinstance(elem, AND_list):
        
        # Default
        final_result = True
       
        # For each elemtn
        for e in elem:
        
            # Evaluate and get result
            result = evaluate_cond_elements(e, input_sent, var_dict)
            
            # Stop if False is found in AND list
            if not result:
                final_result = False
                break

        # If negation, reverse boolean
        if elem.is_negation:
            final_result = reverse_boolean(final_result)
         
        return final_result
        
    # If OR list
    if isinstance(elem, OR_list):
        
        # Default
        final_result = False

        # For each element
        for e in elem:
            
            # Evaluate and get result
            result = evaluate_cond_elements(e, input_sent, var_dict)
                    
            # If negation, reverse boolean
            if elem.is_negation:
                result = reverse_boolean(result)
             
            # Stop if True is found in OR list
            if result:
                final_result = True
                break
            
        # If negation
        if elem.is_negation:
            final_result = reverse_boolean(final_result)
         
        return final_result        
                
    # If string
    if isinstance(elem, str):       
                
        # Evaluate and get result
        result = Condition(elem).evaluate(input_sent, var_dict)
        return result

# ----------------------------------------
### FUNCTIONS for variable constraints handling
# ----------------------------------------
   
def is_valid_variable(X,Y):
    """Given token (X) and variable (Y), make sure token 
    fits variable's constraints."""

    # - Positive constraints: break if mismatch found -
    
    if not(Y.constr_id == [] or (str(X.id) in Y.constr_id)):
        return False
       
    if not (Y.constr_lemma == [] or (X.lemma in Y.constr_lemma)):
        return False
 
    if not (Y.constr_form == [] or (X.form in Y.constr_form)):
        return False
        
    if not (Y.constr_x_pos == [] or (X.x_pos in Y.constr_x_pos)):
        return False
        
    if not (Y.constr_deprel == [] or (X.deprel in Y.constr_deprel)):
        return False    
        
    if not(Y.constr_u_pos == [] or (X.u_pos in Y.constr_u_pos)):
        return False
        
    # - Negative constraints: break if mismatch is found -
    
    if (Y.neg_constr_id != [] and str(X.id) in Y.neg_constr_id):
        return False       
        
    if (Y.neg_constr_lemma != [] and X.lemma in Y.neg_constr_lemma):
        return False
 
    if (Y.neg_constr_form != [] and X.form in Y.neg_constr_form):
        return False
        
    if (Y.neg_constr_x_pos != [] and X.x_pos in Y.neg_constr_x_pos):
        return False
        
    if (Y.neg_constr_deprel != [] and X.deprel in Y.neg_constr_deprel):
        return False    
        
    if (Y.neg_constr_u_pos != [] and X.u_pos in Y.neg_constr_u_pos):
        return False  
        
    return True
    
def extract_elements_from_nested_list(nested_list):
    """Extract elements from nested list."""
    
    new_list = []
    
    # For item in nested list
    for item in nested_list:
        
        # Detangle
        if type(item) != str:            
            new_item = extract_elements_from_nested_list(item)
            new_list += new_item
        else:
            new_list.append(item)
    
    return new_list
     
def matches_constraints(token_set, var_dict):
    """Make sure token set matches constraints of variables."""
    
    # Match up token and variable dict
    matched_up = zip(token_set, var_dict.keys())
        
    # For each pair
    for token, var in matched_up:
        
        # Check whether token works as fill for variable
        var = var_dict[var]
        if not is_valid_variable(token, var):
            return False
            
    return True

def check_exclusive_variables(premise_tokens, excl_var_dict):
    """Check whether exclusive variables match any tokens in the sentence."""
    
    # For each var in exclusive variable dict
    for var in excl_var_dict.keys():  
        
        # Test whether it would fit for any token
        for token in premise_tokens:
            if is_valid_variable(token, excl_var_dict[var]):
                return True
            
    return False