# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 22:23:41 2018

@author: jsuter
"""


from rule_reading_system import *

import classes_for_parsing_results


def test_AND_OR_classes():
    
    string = "pred(X) & pred(Y) | pred(Z)"
    print(OR_list(string))
    print(AND_list(string))
    
    
    

def main():
    
        test_AND_OR_classes()
    
    
    
if __name__ == "__main__":
    main()
