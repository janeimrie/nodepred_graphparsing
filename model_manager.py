from operator import itemgetter
import string
import os
import data_loader as DL
import evaluation as evl
import torch
from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger, MultiTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
from flair.data import Dictionary, Sentence
from flair.hyperparameter.param_selection import SearchSpace, Parameter, SequenceTaggerParamSelector
from hyperopt import hp


class ModelManager:
  
    def __init__(self):
        pass

    def create_model(self, model_name, data_label_type, transformer_embedding_type, hidden_layer_size, crf,  load_premade_label_dict, training_parameters):
        chpc = False
        if chpc:
            corp = "/home/jimrie/lustre/jimrie/Code"
            mod = "/home/jimrie/lustre/jimrie/Code/nodepred_graphparsing"
        #convert parameters
        load_premade_label_dict = True if load_premade_label_dict == "True" else False #can maybe use eval
        crf = True if crf == "True" else False #same as above

        #create corpus

        label_type = data_label_type
        corpus = DL.create_corpus()

        # create label dictionary for a NER tagging task
        if not load_premade_label_dict:
            print("Creating new dictionary")
            ner_dictionary = corpus.make_label_dictionary(label_type=label_type)
            if chpc:
                ner_dictionary.save(f"{corp}/corpus/corpus_label_dict_{label_type}")
            else:
                ner_dictionary.save(f"../corpus/corpus_label_dict_{label_type}")
        else:
            print("Loading old dictionary")
            ner_dictionary = Dictionary
            if chpc:
                ner_dictionary = ner_dictionary.load_from_file(f"{corp}/corpus/corpus_label_dict_{label_type}")
            else:
                ner_dictionary = ner_dictionary.load_from_file(f"../corpus/corpus_label_dict_{label_type}")

        if transformer_embedding_type == "BERT":
            embedding_label = "BERT"
            embeddings = TransformerWordEmbeddings(model = 'SpanBERT/spanbert-base-cased',
                                                fine_tune=True,
                                                subtoken_pooling = 'first_last')

            model = SequenceTagger(hidden_size=int(hidden_layer_size),
                                embeddings=embeddings,
                                tag_dictionary=ner_dictionary,
                                tag_type=label_type,
                                use_crf=crf,
                                use_rnn=False)
        else:
            embedding_label = "Glove"
            embeddings = WordEmbeddings('glove')

            model = SequenceTagger(hidden_size=int(hidden_layer_size),
                                    embeddings=embeddings,
                                    tag_dictionary=ner_dictionary,
                                    tag_type=label_type,
                                    use_crf=crf)

        mbcs = None if training_parameters["Mini Batch Chunk Size"] == "None" else int(training_parameters["Mini Batch Chunk Size"])
        check = True if training_parameters["Checkpoint"] == "True" else False #eval
        plot = True if training_parameters["Plot training curves and weights"] == "True" else False #eval
        self.train_model(model, model_name=model_name, model_corpus= corpus, embedding_type= embedding_label ,learning__rate= float(training_parameters["Learning Rate"]), max__epochs= int(training_parameters["Max epochs"]), mini_batch__size=int(training_parameters["Mini Batch Size"]), mini_batch_chunk__size= mbcs, m_checkpoint= check, plot_training_curves_and_weights= plot)

        return model



    def train_model(self, model, model_name ,model_corpus, embedding_type,learning__rate, max__epochs, mini_batch__size, mini_batch_chunk__size = None, m_checkpoint=True, plot_training_curves_and_weights = True):
        chpc = False
        mod = "/home/jimrie/lustre/jimrie/Code/nodepred_graphparsing"
        trainer = ModelTrainer(model, corpus= model_corpus)

        torch.cuda.empty_cache()

        print("Training model now.....")

        test = False

        if chpc:
            mod_dir = f'{mod}/models/{model_name}'
        else:
            mod_dir = f'models/{model_name}'

        if embedding_type == "BERT":
            embeddings_storage__mode='none'
            print(f"Storing embeddings on {embeddings_storage__mode}")

            if test:
                trainer.fine_tune(mod_dir,
                            learning_rate=learning__rate,
                            mini_batch_size=mini_batch__size,
                             # first try without it, and then if the model doesn't fit on the GPU
                            # try 1 or a number that divides the minibatch_size (e.g. 32 minibatch and 4 minibatch_chunk).
                            mini_batch_chunk_size= mini_batch_chunk__size,
                            embeddings_storage_mode= embeddings_storage__mode,
                            max_epochs= max__epochs ,
                            checkpoint=m_checkpoint,
                            write_weights=plot_training_curves_and_weights)

                if plot_training_curves_and_weights:
                    plotter = Plotter()
                    plotter.plot_training_curves(f'{mod_dir}/loss.tsv')
                    plotter.plot_weights(f'{mod_dir}/weights.txt')


        else:
            embeddings_storage__mode= 'gpu'
            print(f"Storing embeddings on {embeddings_storage__mode}")

            if test:
                trainer.train(mod_dir,
                            learning_rate=learning__rate,
                            mini_batch_size=mini_batch__size,
                            mini_batch_chunk_size= mini_batch_chunk__size,
                            embeddings_storage_mode= embeddings_storage__mode,
                            max_epochs= max__epochs ,
                            checkpoint=m_checkpoint,
                            write_weights=plot_training_curves_and_weights)

            if plot_training_curves_and_weights:
                    plotter = Plotter()
                    plotter.plot_training_curves(f'{mod_dir}/loss.tsv')
                    plotter.plot_weights(f'{mod_dir}/weights.txt')

    
        
    def resume_training(self, path, num_epochs : int):
        # continue training at later point. Load previously trained model checkpoint, then resume
        
        
        experiment = False

        if experiment:
            trained_model = SequenceTagger.load(path + '/checkpoint.pt')

            trainer = ModelTrainer(trained_model, corpus= DL.create_corpus())
            # resume training best model
            trainer.resume(trained_model,
                base_path=path + '-resume',
                max_epochs=num_epochs
                )

    def run_predictions(self, path : string, token_list : list, target_nodes : list):
        print("Performing evaluation...")
        plotter = Plotter()
        plotter.plot_training_curves(f'{path}\loss.tsv')
        plotter.plot_weights(f'{path}\weights.txt')
        print("Plotting loss curves now...")
        #print(os.getcwd())

        experiment = False

        if experiment:
            chpc = False
            print("Running predictions for dev set now")
            if chpc:
                mod_path = f"/home/jimrie/lustre/jimrie/Code/nodepred_graphparsing/models/{path}/final-model.pt"
            else:
                mod_path = "./models/Sur/final-model.pt"
            #print(mod_path)
            trained_model = SequenceTagger.load(model_path = mod_path)
            input = token_list[0]
            sentence = Sentence("Rockwell International Corp's Tulsa unit said it signed a tentative agreement extending its contract with Boeing Co to provide structural parts for Boeing's 747 jetliners")
            
            trained_model.predict(sentence)
            print(sentence)
            #for label in sentence.get_labels():
            #    print(label)
            print(sentence.annotation_layers)
            #print(token_list[0])
            #sentence_list = token_list.copy()
            #trained_model.predict(sentence_list)
            ##initial_list = [item.get_labels('surface') for item in sentence_list]
            #final_predictions_list = [] #[item.value for item in initial_list]
            #for item in sentence_list:
            #    for label in item.get_labels('surface'):
            #        final_predictions_list.append(label.value)
            #print(len(final_predictions_list))
            #print(len(target_nodes))
            #outcomes_surface = evl.run_evaluation(pred = final_predictions_list, targ=target_nodes,label_type='surface')
            #return outcomes_surface

        
    def hyperparameter_tuning(label_type, model_path, epochs, training__runs, embedding, hidden_size, dropout, 
                learning_rate, mini_batch_size):
        
        #https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_8_MODEL_OPTIMIZATION.m 

        # define your search space
        search_space = SearchSpace()
        corpus = DL.create_corpus()

        #Embeddings must always be filled, even if just trialing one
        search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[embedding])
        search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[128, 512, 1024])
        search_space.add(Parameter.DROPOUT, hp.uniform, low=0.01, high=0.1)
        search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2])
        search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 32])

        param_selector = SequenceTaggerParamSelector(corpus,
                                             'ner',
                                             'resources/results',
                                             training_runs=3,
                                             max_epochs=5)

        
