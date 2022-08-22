import json
import model_manager
import sys
import data_loader as DL

# model_spec.txt-path model_index -e -rt

def main():
    models_manager = model_manager.ModelManager()
    print(sys.argv)
    model_specifications_file_path = sys.argv[1]
    eval_ = True if sys.argv[3] == "-e" else False
    retrain = True if sys.argv[4] == "-rt" else False

    if retrain:
        model_path = sys.argv[5]
        epochs = int(sys.argv[6])
        models_manager.resume_training(path = model_path, num_epochs= epochs)

    elif eval_:
        eval_tools = DL.load_token_file("../corpus/dev_tokens")
        dev_list = eval_tools[0]
        target_node = eval_tools[1]
        print(model_specifications_file_path)
        result = models_manager.run_predictions(path = model_specifications_file_path, target_nodes=target_node ,token_list = dev_list)
        print(f"Results: {result}")

    else:
            model_index = int(sys.argv[2])
            with open(model_specifications_file_path,"r") as model_file:
                all_models = model_file.readlines()

                model_dict = json.loads(all_models[model_index])
                print(model_dict)

                model = models_manager.create_model(model_name =  model_dict["Model Name"], data_label_type = model_dict["Label Type"], transformer_embedding_type= model_dict["Embedding Type" ], hidden_layer_size = model_dict["Hidden Layer Size"], load_premade_label_dict= model_dict["Load premade label dictionary"], crf = model_dict["Use CRF"], training_parameters = model_dict["Training Parameters"])
                
                   

if __name__ == "__main__":
    main()


#.\nodepred_graphparsing\.venv\Scripts\Activate.ps1
