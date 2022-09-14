import json
import pickle
import os
from flair.data import Sentence
from flair.datasets import ColumnCorpus
from os import listdir
from os.path import isfile, join


def read_in_data(filename):
    """Takes in a file and returns is a list of JSON objects

    Parameters
    ----------
    filename : String
        Path to file

    Returns
    -------
    [JSON]
        List of JSON object
    """
    with open(filename,"r") as eds_file:
        text = eds_file.readlines()

    json_object_list = [json.loads(item) for item in text]

    return json_object_list

def read_in_tree_data(filename):
    """Read in tree files and returns list of strings

    Parameters
    ----------
    filename : String
        Path to file

    Returns
    -------
    [String]
        List of strings
    """
    with open(filename,"r") as tree_file:
        text = tree_file.readlines()

    return text

def extract_pos(nodelist, number_of_tokens):
    """Extracts EDS surface and abstract node labels given a dictionary 

    Parameters
    ----------
    nodelist : Dictionary
        Dictionary of node names, and label types
    number_of_tokens : Int
        Number of tokens for given sentence
        _description_
    Returns
    -------
    [[Strings]]
        List which contains list of surface, abstract labels and surface span info
    """
    nodes = nodelist['nodes']

    surface_tags = [None] * number_of_tokens
    abstract_tags = [None] * number_of_tokens
    span_info = []

    for node in nodes:
        from_anchor_value = int(node['anchors'][0]['from'])
        end_anchor_value = int(node['anchors'][0]['end'])
        node_label = node['label']
        if 'is_surface' in node:
            temp_list = [from_anchor_value, end_anchor_value, node_label]
            span_info.append(temp_list)
        if from_anchor_value == end_anchor_value:
            if 'is_surface' in node:
                if not surface_tags[from_anchor_value] is None:
                    surface_tags[from_anchor_value] = f"{surface_tags[from_anchor_value]};{node_label}"
                else:
                    surface_tags[from_anchor_value] = node_label
            else:
                if not abstract_tags[from_anchor_value] is None:
                    abstract_tags[from_anchor_value] = f"{abstract_tags[from_anchor_value]};{node_label}"
                else:
                    abstract_tags[from_anchor_value] = node_label

        else:
            for index in range(from_anchor_value, end_anchor_value+1 ):
                if 'is_surface' in node:
                    if not surface_tags[index] is None:
                        surface_tags[index] = f"{surface_tags[index]};{node_label}"
                    else:
                            surface_tags[index] = node_label
                else:
                    if not abstract_tags[index] is None:
                        abstract_tags[index] = f"{abstract_tags[index]};{node_label}"
                    else:
                        abstract_tags[index] = node_label

    return [surface_tags,abstract_tags, span_info]

def extract_tokens(token_list):
    """Extracts tokens from a given sentence 

    Parameters
    ----------
    token_list : [Strings]
        List of tokens 

    Returns
    -------
    _[Strings]
        List of string representations of tokens for a sentence
    """
    token_dictionaries = token_list['tokens']
    tokens = []

    #Can maybe make more efficient with a list comprehension
    for sub_dict in token_dictionaries:
        tokens.append(sub_dict['form'].strip(",."))

    return tokens

def extract_linearised_tree_labels(filepath):
    """Extracts tree2label labesl for .seq_lu files

    Parameters
    ----------
    filepath : String
        Path to file

    Returns
    -------
    [Strings]
        List of all tags in the file
    """
    path_ = filepath
    with open(path_,"r") as labelfile:
        lines = labelfile.readlines()
        tags =[]
        contained_tags = []
        temp_list = []
        for line in lines:
            if "-EOS-" in line:
                contained_tags.append(temp_list)
                temp_list = []
                continue
            elif line == '\n':
                tags.append(" ")
                continue
            elif "-BOS-" in line:
                continue
            else:
                temp = line.split()
                if len(temp) > 1:
                    temp_tag = temp[2].strip('\n')
                    tags.append(temp_tag)
                    temp_list.append(temp_tag)
    return tags
   
def add_linearised_labels_to_binary_file(labels, label_type):
    """Helper method to add seq2tree labels 

    Parameters
    ----------
    labels : [Strings]
        List of string representations of labels
    label_type : String
        Label type for the different sets - either dev, test or train
    """
    filepath = f"../corpus/{label_type}_abstract_labels"
    with open(filepath,"wb") as output:
        pickle.dump(labels, output)    

def add_linearised_labels_to_textfile(filename, label_list, label_type):
    """ Helper method to add tree2label labels to file

    Parameters
    ----------
    filename : String
        File to which labels need to be added
    label_list : [Strings]
        Lists of labels
    label_type : String
        Either linearised_labels or surface_ner
    """

    add_linearised_labels_to_binary_file(labels=label_list, label_type=label_type)

    with open(f"../corpus/{label_type}.txt", 'w') as outputfile:
        with open(filename, "r") as inputfile:
            input = inputfile.readlines()
            new_list= [*zip(input, label_list)]#, bio_list)]
            joined_list = [' '.join(map(str, item)).replace('\n',"") for item in new_list]
        for item in joined_list:
            outputfile.write(item)
            outputfile.write("\n")
   
def create_bio_tags_for_surface_tokens(label_list):
    """Assigns a B/I/O tag to a surface token label

    Parameters
    ----------
    label_list : [Strings]
        List of labels for each token

    Returns
    -------
    [Strings]
        List of tagged token labels
    """
    surf = label_list
    tags = []
    for index in range(0, len(surf)):
        tag_current = surf[index]
        if surf[index] is not None: #no tag for token at that particular index
            if index > 0:
                if surf[index-1] is not None:
                    tag_prev = surf[index-1]
                    if ";" not in tag_current and ";" not in tag_prev: #not overlapping span
                        if  tag_prev== tag_current:
                            surf_bio_tag = f"X-{tag_current}"
                        else:
                            surf_bio_tag = f"A-{tag_current}"
                    else: # potential overlapping span
                        surf_bio_tag = ""
                        list_of_current_tags = []
                        list_of_prev_tags = []

                        if ";" in tag_current:
                            list_of_current_tags = tag_current.split(";")
                        
                        if ";" in tag_prev:
                            list_of_prev_tags = tag_prev.split(";")

                        if len(list_of_prev_tags)== 0 and len(list_of_current_tags) > 0: #prev does not have multiple tags and current does                 
                            for tag in list_of_current_tags:
                                if tag_prev == tag:
                                    surf_bio_tag = surf_bio_tag + ";" +f"X-{tag}"
                                else:
                                    surf_bio_tag = surf_bio_tag + ";" +f"A-{tag}"
                                    
                        elif len(list_of_prev_tags)> 0 and len(list_of_current_tags) == 0: #prev has multiple tags and current doesn't
                            if tag_current in list_of_prev_tags:
                                surf_bio_tag = f"X-{tag_current}"
                            else:
                                surf_bio_tag = f"A-{tag_current}"

                        else: #both prev and current have multiple tags each 
                            for current in list_of_current_tags:
                                for prev in list_of_prev_tags:
                                    if current == prev:
                                        surf_bio_tag = surf_bio_tag+ ";" +f"X-{current}"
                                        break
                                else:
                                    surf_bio_tag = surf_bio_tag + ";" +f"A-{current}"                   
                else:
                    surf_bio_tag = ""
                    if ";" not in tag_current:
                        surf_bio_tag = f"A-{tag_current}"
                    else:
                        current_tag_list = tag_current.split(";")
                        for current in current_tag_list:
                            surf_bio_tag = surf_bio_tag + ";" +f"A-{current}"
            else:
                surf_bio_tag = f"A-{tag_current}"

        else:
            surf_bio_tag = f"D-{tag_current}" #assigned None label for network training purposes

        surf_bio_tag = surf_bio_tag.strip(";")
        tags.append(surf_bio_tag)

    return tags

def create_gold_spans(tokens, spans):
    """ Creates the gold spans for a set of tokens

    Parameters
    ----------
    tokens : [[Strings]]
        List of tokenised sentences
    spans : [[Strings]]
        Unformatted span information

    Returns
    -------
    list
        Formatted spans with token values
    """
    for sublist in spans:
        start = sublist[0]
        end = sublist[1]
        token = ""
        if start == end:
            token = tokens[start]
            sublist.append(token)
        else:
            for index in range(start,end+1):
               token = token + " " + tokens[index]
            sublist.append(token.strip())
    return spans

def load_data_into_textfile(input_file, output_file, token_file, token__list, surf_node_list, surf_tag_list ,last ):
    inputfile=input_file
    outputfile = output_file

    tokenfile = token_file
    #token_list = []
    #surface_nodes = []
    input = read_in_data(filename=inputfile)
    with open(outputfile , "a") as traindata_outputfile:
        with open(tokenfile, "wb") as tokens_outputfile:
            counter = 0
            for item in input:
                toks = extract_tokens(item)
                #token_list.append(toks)
                token__list.append(toks)
                
                pos = extract_pos(item, len(toks))
                surf = pos[0]
                abstr = pos[1]
                spans = pos[2]
                #print(spans)
                final_spans = create_gold_spans(toks, spans)
                surf_node_list.append(final_spans)
                #print(final_spans)
                surf_bio = create_bio_tags_for_surface_tokens(surf)  
                surf_tag_list.extend(surf_bio) #--orig
                #surface_nodes.extend(surf)
                
                #result = result + surf
                  

                # For debugging
                if len(toks) != len(surf_bio) or  len(toks) != len(abstr):
                    print(f"{counter}. Sentence: {item['input']}, Toks : {len(toks)}, Abstr: {len(abstr)}, Surf : {len(surf)}")
                    print(pos)
                    print(surf)
                    print(abstr)
                    print("\n")
                    ##################
                else:
                    for index in range(0, len(toks)):
                        traindata_outputfile.write(f"{toks[index]} {surf[index]};{abstr[index]} {surf[index]} {abstr[index]} {surf_bio[index]}\n")
                        if index == (len(toks)-1):
                            traindata_outputfile.write("\n")
                counter+=1
            #print(len(token_list))
            #last = True #REMOVE LATER 
            if last:
                #token_list.append(surf_node_list)
                #token_list.append(surface_nodes) #HERE for evaluation - makes it easier
                #pickle.dump(token_list, tokens_outputfile)
                token__list.append(surf_node_list)
                token__list.append(surf_tag_list)
                pickle.dump(token__list, tokens_outputfile)

def load_tree_data_into_textfile(input_file, output_file):
    inputfile=input_file
    outputfile = output_file
    input = read_in_tree_data(inputfile)
    with open(outputfile , "a") as treedata_outputfile:
        for item in input:
            treedata_outputfile.write(item)
      
def create_corpus():
    # define columns
    columns = {0: 'token', 1:'both' ,2: 'surface', 3: 'abstract', 4:'surface_ner', 5:'linearised_labels', 6: 'abstract_ner'} 

    # this is the folder in which train, test and dev files reside
    data_folder = "C:/Users/LENOVO/Documents/Honours/Proj/Code/corpus"

    # init a corpus using column format, data folder and the names of the train, dev and test files
    # place into try-catch block to test for if text files actually exist 
    corpus  = ColumnCorpus(data_folder, columns, 
                                train_file='train.txt',
                                test_file='test.txt',
                                dev_file='dev.txt')


    return corpus

def add_all_sets_to_text_file(path, output_file, token_file):
    """ Main method for adding data to textfile in ColumnCorpus format for Flair

    Parameters
    ----------
    path : String
        Path to file that data must be read from
    output_file : String
        Output file name
    token_file : String
        File containing tokens to be added to output file
    """

    file_names = [file_name for file_name in listdir(path) if isfile(join(path, file_name))]
    last = False
    tokenlist = []
    targlist = []
    tag_gold_list = []
    for i in range(len(file_names)):
        file_name = file_names[i]
        if i >= (len(file_names)-1):
            last = True
        result = load_data_into_textfile(f"{path}/{file_name}", output_file, token_file, token__list=tokenlist, surf_node_list=targlist, surf_tag_list= tag_gold_list, last=last)

def add_all_tree_sets_to_text_file(path, output_file):
    """ Main method that adds trees to file according to dev-test-train split

    Parameters
    ----------
    path : String
        Path to file
    output_file : String
        Filename that trees must be outputted to
    """
    file_names = [file_name for file_name in listdir(path) if isfile(join(path, file_name))]
    for file_name in file_names:
        load_tree_data_into_textfile(f"{path}/{file_name}",  output_file)

def load_token_file(path):
    """ Loads a binary file that contains surface tokens

    Parameters
    ----------
    path : String
        Path to token file

    Returns
    -------
    _type_
        _description_
    """
    list_of_lists = pickle.load(open(path,'rb'))
    #print(list_of_lists)
    surface_nodes = list_of_lists[len(list_of_lists)-2]
    del list_of_lists[len(list_of_lists)-2]
    #print(surface_nodes)
    surface_tags = list_of_lists[len(list_of_lists)-1]
    del list_of_lists[len(list_of_lists)-1]
    
    sentence_list = convert_stringlist_to_sentences(list_of_lists)
    return [sentence_list, surface_nodes, surface_tags]

def load_abstract_token_text_file(path):
    token_file = open(path, "r")
    lists = token_file.readlines()
    return lists

def load_abstract_token_binary_file(path):
    """ Loads in a binary file containing tokens

    Parameters
    ----------
    path : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    lists = pickle.load(open(path, 'rb'))
    return lists

def convert_stringlist_to_sentences(input_sentence_list):
    """Converts a list of strings into a list of Sentences

    Parameters
    ----------
    input_sentence_list : [Strings]
        List of strings 

    Returns
    -------
    [Sentences]
        List of Sentence objects
    """
    sentences = [Sentence(item) for item in input_sentence_list]
    return sentences

def main():
    paths = [ "../data/extracted/train" , "../data/extracted/test" , "../data/extracted/dev" ]
    output_files_paths = ["../corpus/_train.txt", "../corpus/_test.txt", "../corpus/_dev.txt"]
    output_token_files_paths = ["../corpus/train_tokens","../corpus/test_tokens","../corpus/dev_tokens"]

    tree_paths = [ "../data/extracted/trees/train_tree" , "../data/extracted/trees/test_tree" , "../data/extracted/trees/dev_tree" ]
    tree_output_files_paths = ["../corpus/trees/train.txt", "../corpus/trees/test.txt", "../corpus/trees/dev.txt"]

    recreate_files = True
    if recreate_files:
        for input_directory,output_file, token_file in zip(paths,output_files_paths, output_token_files_paths):
            add_all_sets_to_text_file(input_directory, output_file, token_file)

    create_trees = True
    if create_trees:
        for input, output in zip(tree_paths, tree_output_files_paths):
            add_all_tree_sets_to_text_file(input,output)

    add_linear_labels = True
    if add_linear_labels:
        linearised_file_paths = ["../corpus/trees/Redwoods-train.seq_lu","../corpus/trees/Redwoods-test.seq_lu","../corpus/trees/Redwoods-dev.seq_lu"]
        label_types = ["train","test","dev"] 
        for file_path, label_type in zip(linearised_file_paths, label_types):
            #label_lists = extract_linearised_tree_labels(file_path)
            labl_list = extract_linearised_tree_labels(file_path)
            #contained_list = label_lists[1]
            add_linearised_labels_to_textfile(f"../corpus/_{label_type}.txt",labl_list,label_type, contained_label_list=None)

if __name__ == "__main__": 
    main()













