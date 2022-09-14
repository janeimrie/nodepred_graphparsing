from nltk.tree import Tree
import sys
import codecs
import time
from tree import SeqTree, RelativeLevelTreeEncoder
#from tree2 import SeqTree, RelativeLevelTreeEncoder

def getTreeSpans(t, spans : list, start_index=0,):
    """Gets spans for a tree
    Created by Jan Buys
    Modified by Jane Imrie

    Parameters
    ----------
    t : tree
        String representation of tree
    spans : list
        List of spans for given input
    start_index : int, optional
        Index for iterating through tree, by default 0

    Returns
    -------
    _type_
        _description_
    """
    #print(type(span))
    #spans = span
    if t.height() == 2:
        return 1
    span_size = 0
    for i, child in enumerate(t):
        st_size = getTreeSpans(child, spans ,start_index + span_size)
        span_size += st_size
    labels = t.label().split(';')
    for l in labels:
        if l != 'S':
            spans.append((l, start_index, start_index+span_size-1))
    return span_size

def trees_to_spans(trees):
    result_list = []
    for tree in trees:
        try:
            tree = Tree.fromstring(tree, remove_empty_top_bracketing=True)
        except ValueError: #error occurs when sentences doesn't start with capital letter - just add extra )
            tree = tree +")"
            tree = Tree.fromstring(tree, remove_empty_top_bracketing=True)
        #tree.pretty_print() - uncomment to have tree print out to console in format
        spans = []

        getTreeSpans(t = tree, spans = spans, start_index= 0)

        result_list.append(spans)
    return result_list

def sequence_to_parenthesis(sentences,labels):
    """
        Transforms a list of sentences and predictions (labels) into parenthesized trees
        @param sentences: A list of list of (word,postag)
        @param labels: A list of list of predictions
        @return A list of parenthesized trees

        Code sourced from: https://github.com/aghie/tree2labels
    """
    
    parenthesized_trees = []  
    relative_encoder = RelativeLevelTreeEncoder()
    
    f_max_in_common = SeqTree.maxincommon_to_tree
    f_uncollapse = relative_encoder.uncollapse
    
    total_posprocessing_time = 0
    for noutput, output in enumerate(labels):       
        if output != "": #We reached the end-of-file
            init_parenthesized_time = time.time()
            sentence = []
            preds = []
            for ((word,postag), pred) in zip(sentences[noutput][1:-1],output[1:-1]):
                        
                if len(pred.split("_"))==3: #and "+" in pred.split("_")[2]:
                    sentence.append((word,pred.split("_")[2]+"+"+postag))             
                              
                else:
                    sentence.append((word,postag)) 
                
                #TODO: This is currently needed as a workaround for the retagging strategy and sentences of length one
#                 if len(output)==3 and output[1] == "ROOT":
#                     pred = "NONE"     
                
                preds.append(pred)
            tree = f_max_in_common(preds, sentence, relative_encoder)
                        
            #Removing empty label from root
            if tree.label() == SeqTree.EMPTY_LABEL:
                
                #If a node has more than two children
                #it means that the constituent should have been filled.
                if len(tree) > 1:
                    print ("WARNING: ROOT empty node with more than one child")
                else:
                    while (tree.label() == SeqTree.EMPTY_LABEL) and len(tree) == 1:
                        tree = tree[0]

            #Uncollapsing the root. Rare needed
            if "+" in tree.label():
                aux = SeqTree(tree.label().split("+")[0],[])
                aux.append(SeqTree("+".join(tree.label().split("+")[1:]), tree ))
                tree = aux
            tree = f_uncollapse(tree)
            

            total_posprocessing_time+= time.time()-init_parenthesized_time
            #To avoid problems when dumping the parenthesized tree to a file
            aux = tree.pformat(margin=100000000)
            parenthesized_trees.append(aux)

    return parenthesized_trees 
