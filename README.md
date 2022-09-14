# nodepred_graphparsing

This repo contains the experiment for Node Prediction in Meaning Representation Parsing 

This code facilitates the creation of Flair SequenceTagger models with either Glove or BERT embeddings.

Data for these experiments is sourced from this repository: https://gitlab.cs.uct.ac.za/jbuys/mrs-processing (accessible with UCT staff and student emails). Run the following line once the repo has been cloned and the data has been downloaded:
    python src/extract-convert-mrs.py --deepbank --eds -i data/original/erg1214/tsdb/gold/ -o data/extracted/ --convert_semantics --extract_semantics --extract_semantic_trees 2> data/extracted/all.err

This will extract the EDS data and convert it to tree form.

Then run the dataset.py command from this repository: https://github.com/aghie/tree2labels, with the os and encode unaries flags enabled. 

Then, the data files need to formatted into a ColumnCorpus format and amalgamated:
    mkdir corpus
    python3 data_loader.py

To create models, create a .txt file with the model parameters in it. An example of this, model_specs.txt, has been included. 

    Example: 
    # Surface Tokens + Glove + No CRF
    {"Model Name" : "Surface Tokens + Glove+ No CRF", "Embedding Type" : "Glove", "Label Type" : "surface" , "Hidden Layer Size": "256", "Use CRF": "False", "Output path":"./models/model_", "Load premade label dictionary" : "False" ,"Training Parameters":{"Learning Rate" : "0.1" , "Max epochs": "30" , "Mini Batch Size": "10", "Mini Batch Chunk Size" : "None", "Checkpoint" : "True", "Plot training curves and weights" : "False"}}

The first line, the # followed by the model name, is optional. 

To create a model and initiate training, (example):

    python3 test.py model_specs.txt 9 -- -- -- > output_Abstract_Tokens_Glove_No_CRF.txt

Note that "9" refers to the line in the text file that the model information is found. Indexing starts at 0. 

To resume training/do more epochs, (example):

    python3 test.py "Abstract Tokens + Glove + No CRF" -- -- -rt -- 20 >> retrain_output_Abstract_Tokens_Glove_No_CRF.txt

Here, the 20 refers to the new max epochs value and the -rt flag is to indicate that you want additional training. 

To find the optimal learning rate:

    python3 test.py "Surface BIO + Glove+ No CRF" -l -- -- -tn 10 8 "surface_ner" 0 "Glove" > /home/jimrie/lustre/jimrie/Code/nodepred_graphparsing/models/tuning_LR_Surface_BIO_Glove_No_CRF.txt

The 10 is for the number of iterations and 8 is for the batch size. "surface_ner" refers to batch size and "Glove" to the embedding type. 5 is for the number of epochs. 

To perform an evaluation:

    python test.py "Surface BIO + BERT + No CRF" -- -e -- -- "surface_ner"

Either "surface_ner" or "linearised_labels" can be given as input, depending on the model being created

This code was run on the CHPC (https://www.chpc.ac.za/) and requires at least 1 GPU and 20 CPUS to run. 








