# This is the main driver class for the experiments
# It accepts command line input

import json
import model_manager
import sys
import data_loader as DL
import utils as utils

# model_spec.txt-path model_index -e -rt

def main():
    models_manager = model_manager.ModelManager()
    print(sys.argv)
    model_specifications_file_path = sys.argv[1]
    eval_ = True if sys.argv[3] == "-e" else False
    retrain = True if sys.argv[4] == "-rt" else False
    tune = True if sys.argv[5] == "-tn" else False

    if retrain:
        model_path = sys.argv[1]
        epochs = int(sys.argv[6])
        models_manager.resume_training(path = model_path, num_epochs= epochs)

    elif tune:
        model_path = sys.argv[1]
        dropout = True if sys.argv[2] == "-d" else False
        learning = True if sys.argv[2] == "-l" else False
        both_drop_learn = True if sys.argv[2] == "-dl" else False
        iterations = int(sys.argv[6])
        mini_batchsize = int(sys.argv[7])
        tag_type = sys.argv[8]
        epoch = int(sys.argv[9])
        embeddings = sys.argv[10]
        if both_drop_learn:
            models_manager.find_opt_learning_rate(mod_path=model_path, itrs=iterations, mini_b_size=mini_batchsize)
            models_manager.hyperparameter_tuning(model_path = model_path,tagtype=tag_type, epochs = epoch, embedding = embeddings)
        elif learning:
            models_manager.find_opt_learning_rate(mod_path=model_path, itrs=iterations, mini_b_size=mini_batchsize)
        elif dropout:
            models_manager.hyperparameter_tuning(model_path = model_path,tagtype=tag_type, epochs = epoch, embedding = embeddings)
        else:
            print("Error! No tuning method selected!")


    elif eval_:
        tag_type = sys.argv[6]
        dev_eval_tools = DL.load_token_file("../corpus/test_tokens")
        sentences = dev_eval_tools[0]
        if tag_type == "surface_ner":
            gold_nodes = dev_eval_tools[1]
            gold_tags = dev_eval_tools[2]
            models_manager.run_predictions(path = model_specifications_file_path, target_nodes=gold_nodes ,token_list = sentences, target_tags= gold_tags , label_type=tag_type)
        elif tag_type == "linearised_labels":
            corpus = DL.create_corpus()
            test= corpus.test
            tokens_list = [sentence.get_labels("linearised_labels") for sentence in test]
            gold_labels = [*map(utils.get_tags, tokens_list)]
            gold_sentences = [*map(utils.get_abstract_token, test)]
            abstract_eval_tags = DL.load_abstract_token_binary_file("../corpus/test_abstract_labels")
            #unformatted_abstract_trees = DL.load_abstract_token_text_file("../corpus/trees/test.txt") - UNCOMMENT when sequence_to_trees method is fixed in tree2label code
            abstract_trees = [gold_sentences, gold_labels] #as above comment [item.strip("\n") for item in unformatted_abstract_trees]
            models_manager.run_predictions(path = model_specifications_file_path, target_nodes= abstract_trees, token_list= sentences, target_tags=abstract_eval_tags, label_type=tag_type )


    else:
            model_index = int(sys.argv[2])
            with open(model_specifications_file_path,"r") as model_file:
                all_models = model_file.readlines()
                model_dict = json.loads(all_models[model_index])
                models_manager.create_model(model_name =  model_dict["Model Name"], data_label_type = model_dict["Label Type"], transformer_embedding_type= model_dict["Embedding Type" ], hidden_layer_size = model_dict["Hidden Layer Size"], load_premade_label_dict= model_dict["Load premade label dictionary"], crf = model_dict["Use CRF"], training_parameters = model_dict["Training Parameters"])
                
                   

if __name__ == "__main__":
    main()


#.\nodepred_graphparsing\.venv\Scripts\Activate.ps1