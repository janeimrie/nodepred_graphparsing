from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
import data_loader as DL
import torch

class ModelManager:
  
    def __init__(self):
        self.models = []

    def create_model(self, transformer_embedding_type, hidden_layer_size, crf, model_name, outputfile_path):

        #create corpus 

        label_type = 'label'
        corpus = DL.create_corpus()

        # create label dictionary for a NER tagging task
        ner_dictionary = corpus.make_label_dictionary(label_type=label_type)

        if transformer_embedding_type:
            embedding_label = "BERT"
            embeddings = TransformerWordEmbeddings('spanbert-large-cased')
        else:
            embedding_label = "Glove"
            embeddings = WordEmbeddings('glove')
      
        model = SequenceTagger(hidden_size=hidden_layer_size,
                                embeddings=embeddings,
                                tag_dictionary=ner_dictionary,
                                tag_type=label_type,
                                use_crf=crf)

        new_model_specifications = {"Model Name": model_name, "Model Object": model, "Model Corpus": corpus ,"Model Parameters": {"Embedding Type": embedding_label,"Hidden size": hidden_layer_size, "CRF": crf}}
        self.models.append(new_model_specifications)
        print(f"Created model:{model_name}")
        #with open(outputfile_path , "a") as model_outputfile:
        #    print(f"Writing to outputfile:{outputfile_path}")
        #    model_outputfile.write()


    def train_model(self, model_name, model_output_path, learning__rate, max__epochs, mini_batch__size, mini_batch_chunk__size = None, m_checkpoint=True, plot_training_curves_and_weights = True):

        # NB that the write_weights .train needs to be set based on the value of visualise
        # Make below line more efficient with filter() 
        for model_set in self.models:
            if model_set["Model Name"] == model_name:
                current_model = model_set
                print(f"Retrieving {model_set['Model Name']} with various parameters, preparing for training.....")
                break

        trainer = ModelTrainer(current_model["Model Object"], current_model["Model Corpus"])

        torch.cuda.empty_cache()

        print("Training model now.....")

        if current_model["Model Parameters"]["Embedding Type"] == "BERT":
            embeddings_storage__mode='none'
            print(f"Storing embeddings on {embeddings_storage__mode}")
        else:
            embeddings_storage__mode= 'gpu'
            print(f"Storing embeddings on {embeddings_storage__mode}")

        trainer.train(f'models/{current_model["Model Name"]}',
                    learning_rate=learning__rate,
                    mini_batch_size=mini_batch__size,
                    mini_batch_chunk_size= mini_batch_chunk__size,
                    embeddings_storage_mode= embeddings_storage__mode,
                    max_epochs= max__epochs , 
                    checkpoint=m_checkpoint,
                    write_weights=plot_training_curves_and_weights)

        if plot_training_curves_and_weights:
            plotter = Plotter()
            plotter.plot_training_curves('loss.tsv')
            plotter.plot_weights('weights.txt')
    
        
    def resume_training(self, model_name):
        pass
        
    def hyperparameter_tuning():
        pass
