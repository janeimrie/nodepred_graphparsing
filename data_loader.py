import json
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from os import listdir
from os.path import isfile, join


def read_in_data(filename):
    with open(filename,"r") as eds_file:
        text = eds_file.readlines()


    #for item in range(0, len(text)):
    #    temp = json.loads(text[item])
    #    text[item] = temp

    #More efficient version
    json_object_list = [json.loads(item) for item in text]

    return json_object_list

def extract_pos(nodelist, number_of_tokens):
    nodes = nodelist['nodes']

    pos_tags = [None] * number_of_tokens

    for node in nodes:
        from_anchor_value = int(node['anchors'][0]['from'])
        end_anchor_value = int(node['anchors'][0]['end'])
        node_label = node['label']
        
        if from_anchor_value == end_anchor_value:
            if not pos_tags[from_anchor_value] is None:
                new_label = f"{pos_tags[from_anchor_value]};{node_label}"
                pos_tags[from_anchor_value] = new_label
            else:

                    pos_tags[from_anchor_value] = node_label

        else:
            for index in range(from_anchor_value, end_anchor_value+1 ):
                if not pos_tags[index] is None:
                    new_label = f"{pos_tags[index]};{node_label}"
                else:
                    new_label = node_label
                pos_tags[index] = new_label

    return pos_tags


def extract_tokens(token_list):
    token_dictionaries = token_list['tokens']
    tokens = []

    for sub_dict in token_dictionaries:
        tokens.append(sub_dict['form'].strip(",."))

    return tokens
    

# Note - Edit such that inputfile is piped in and then data is ADDED to train.txt, rather than overwrite train.txt
def load_data_into_textfile(input_file, output_file):
    inputfile=input_file
    outputfile = output_file

    '''toks = extract_tokens(input[396])
    pos = extract_pos(input[396], len(toks))

    print(len(toks))
    print(len(pos))

    print(toks)
    print(pos)'''
    input = read_in_data(filename=inputfile)
    with open(outputfile , "a") as traindata_outputfile:
        counter = 0
        for item in input:
            toks = extract_tokens(item)
            pos = extract_pos(item, len(toks))

            # For debugging
            if len(toks) != len(pos):
                print(f"{counter}. Sentence: {item['input']}, Toks : {len(toks)}, PoS: {len(pos)}")
                print(pos)
                print(toks)
                print("\n")
                ##################
            else:
                for index in range(0, len(toks)):
                    bio_tag = "TAG"
                    if pos[index] is None:
                        bio_tag = "O"
                    else:
                        bio_tag = "B"
                    traindata_outputfile.write(f"{toks[index]} {pos[index]} {bio_tag}\n")
                    if index == (len(toks)-1):
                        traindata_outputfile.write("\n")
            counter+=1
            #traindata_outputfile.write("\n")


def create_corpus():
    # define columns
    columns = {0: 'text', 1: 'label', 2: 'ner'}

    # this is the folder in which train, test and dev files reside
    data_folder = './corpus/'

    # init a corpus using column format, data folder and the names of the train, dev and test files
    # place into try-catch block to test for if text files actually exist
    corpus  = ColumnCorpus(data_folder, columns,
                                train_file='train.txt',
                                test_file='test.txt',
                                dev_file='dev.txt')


    return corpus


def add_all_sets_to_text_file(path, output_file):

    file_names = [file_name for file_name in listdir(path) if isfile(join(path, file_name))]
    for file_name in file_names:
        load_data_into_textfile(f"{path}/{file_name}", output_file)


paths = [ "./data/extracted/train" , "./data/extracted/test" , "./data/extracted/dev" ]
output_files_paths = ["./corpus/train.txt", "./corpus/test.txt", "./corpus/dev.txt"]

for input_directory,output_file in zip(paths,output_files_paths):
    add_all_sets_to_text_file(input_directory, output_file)

#add_all_sets_to_text_file(paths[1], output_files_paths[1])










