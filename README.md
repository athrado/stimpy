# _STIMPY_
# Sentence Transformative Inference Mapping for Python

![](https://github.com/athrado/stimpy/tree/master/Report/tree.png)

_STIMPY_ is an Natural Language Inference engine for inferring whether a hypothesis _h_ follows from a given premise _p_ - or a set of premises.

```
Premise:      Mrs. Norris is a cat.
Hypothesis:   Mrs. Norris is not a poodle.
---
Relation: forward entailment
Valid inference
```

```
Premise 1:   If it is raining, you can't go by bicycle.
Premise 2:   You can go by bicycle.
Hypothesis:  It is raining.
--- 
Relation: negation
Invalid inference
```

_STIMPY_ proves whether a set of premises entails a hypothesis by continuous generation of derived facts through alteration of the premises according to a manually engineered rule set. As derived facts are generated, their logical relation with the premise is tracked and
adjusted as necessary. Thus, when a derived fact is found that matches the hypothesis, the final relation can be used to determine whether the inference is valid, invalid or unknown.

_STIMPY_ was inspired by NaturalLI [(Angeli et Manning, 2014)](https://www.aclweb.org/anthology/D14-1059), yet differs from that system in three significant aspects:

- _STIMPY_ incorporates a formal syntax for writing and maintaining inference rules in an accessible way
- _STIMPY_ aims at producing syntactically correct derived facts
- _STIMPY_ is capable of processing multi-premise problems, while NaturalLI only addresses single-premise problems

Syntax example for hypernym replacement rule:

```
exists_hypernym(X) --> replace(X, hypernym(X))
# relation = forward entailment
```
For more details about the engine, the rule syntax and system performance, please consult the [report](https://github.com/athrado/stimpy/blob/master/Report/FM_JuliaSuter2019.pdf). 


## How to Run

For processing a single NLI problem, specify hypothesis/query and premise set in the main function of `RUN_stimpy.py.` and run with Python.

For evaluating the system on the FraCaS test suite, run `EVALUATE_Fracas.py.` At the beginning of the script, you can select specific sets of FraCaS problems and adjust the timeout. 

For evaluating the development samples, run `EVALUATE_dev_samples.py.`  This script allows you to test test any other set of NLI problems, as long as they are saved in the following format:

```
# Comment line
Premises:
- premise 1
- premise 2
Hypothesis: hypothesis
Relation: [equivalence, forward entailment, reverse entailment, negation, alternation, cover, unknown]
Inference validity: [valid, invalid, unknown]
```

## Rule Extension

To adjust or extend the rule set, modify the text files containing the single-premise and multi-premise rules. There is no need to alter any scripts. Refer to Chapter 2 in the  [report](https://github.com/athrado/stimpy/blob/master/Report/FM_JuliaSuter2019.pdf) in order to create well-formed, meaningful and efficient rules. 

## Contact

If you have any questions, please do not hesitate to contact me by [email](https://user-images.githubusercontent.com/42718928/60554359-d4f49080-9cfc-11e9-92c9-e81852790ff8.png). 