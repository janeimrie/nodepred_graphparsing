import model_manager

models_manager = model_manager.ModelManager()

mod_name = "BERT ; No CRF"

models_manager.create_model(transformer_embedding_type= True, hidden_layer_size= 256, crf = False, model_name=mod_name, outputfile_path="./models/models.txt") 

models_manager.train_model(mod_name, True)