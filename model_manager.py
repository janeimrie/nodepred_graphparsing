import string
import data_loader as DL
import evaluation as evl
import torch
import copy
import utils as utils
from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
from flair.data import Dictionary
from flair.hyperparameter.param_selection import SearchSpace, Parameter, SequenceTaggerParamSelector
from torch.optim.adam import Adam
from hyperopt import hp


class ModelManager:
  
    def __init__(self):
        pass

    def create_model(self, model_name, data_label_type, transformer_embedding_type, hidden_layer_size, crf,  load_premade_label_dict, training_parameters):
        """Creates a Flair SequenceTagger model object with given parameters

        Parameters
        ----------
        model_name : String
            Name for model
        data_label_type : String
            Type of concept being predicted - surface or abstract
        transformer_embedding_type : String
            Embedding type - either Glove or BERT
        hidden_layer_size : Int
            Number of hidden layers
        crf : String
            Either Yes/No for presence of CRF layer
        load_premade_label_dict : Boolean
            True for if want to load a premade dictionary stored on disk
        training_parameters : Dictionary
            Training parameters for the model - including learning rate, batch size, etc

        Returns
        -------
        SequenceTagger
            Model object created using the parameters
        """
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
        """Method used for training model. Parameters are used to specify training regime.

        Parameters
        ----------
        model : SequenceTagger
            Model object created using create_model() 
        model_name : String
            Descriptive name for model
        model_corpus : Corpus
            Corpus containing formatted data for model
        embedding_type : String
            Either BERT or Glove
        learning__rate : float 
            Learning rate for model
        max__epochs : int
            Max amount of epochs for model during training, early stopping employed
        mini_batch__size : int
            Batch size for model
        mini_batch_chunk__size : _type_, optional
            Batch chunk size for model, by default None
        m_checkpoint : bool, optional
            Enable to allow model training to be resumed, by default True
        plot_training_curves_and_weights : bool, optional
            Enable to allow a file of model weights to be created, which can then be plotted, by default True
        """
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
        """Resume training for a model

        Parameters
        ----------
        path : String
            path to model folder that contains checkpoint.pt file
        num_epochs : int
            New maximum number of epochs for the given model. Training epochs is now difference between previous max epochs and new max epochs
        """
        # continue training at later point. Load previously trained model checkpoint, then resume
        print(f"Resuming traing for {num_epochs} epochs")
        
        experiment = False
        trained_model = SequenceTagger.load(path + "/checkpoint.pt")

        if experiment:
            trainer = ModelTrainer(trained_model, corpus= DL.create_corpus())
            # resume training best model
            trainer.resume(trained_model,
                base_path=path + '-resume',
                max_epochs=num_epochs
                )

    def run_predictions(self, path : string, token_list : list, target_nodes : list, target_tags : list, label_type):
        """ Method for getting predictions from a model given a test bed

        Parameters
        ----------
        path : string
            Path to model
        token_list : list
            List of sentences that have been tokenised
        target_nodes : list
            List of gold spans
        target_tags : list
            List of gold tags 
        label_type : String
            Label type for the model, either surface_ner or linearised_labels
        """
        print("Performing evaluation...")

        experiment = True

        if experiment:
            chpc = False
            #print(target_nodes)
            print("Running predictions for dev set now")
            print(path)
            if chpc:
                mod_path = f"{path}/final-model.pt"
            else:
                mod_path = path
            #print(mod_path)
            trained_model = SequenceTagger.load(model_path = mod_path)
            sentence_list = copy.deepcopy(token_list)
            trained_model.predict(sentence_list)
            initial_predicted_list = [item.get_labels() for item in sentence_list]
            if label_type =="surface_ner":
                predicted_spans = []
                outside_counter = 0
                predicted_tags = []
                for sublist in initial_predicted_list:
                    counter = 0
                    temp_list = []
                    sentence = sentence_list[outside_counter]
                    for token in sublist:
                        #print(token)
                        predicted_tags.append(token.value)
                        tag_list = evl.strip_bio_tags(token.value).split(";")
                        if len(tag_list) == 1:
                            tag = tag_list[0]
                            if tag != "None":
                                tup = [counter, counter, tag , sentence[counter].text]
                                temp_list.append(tup)
                        else:
                            for multi_tag in tag_list:
                                tup = [counter, counter, multi_tag , sentence[counter].text]
                                temp_list.append(tup)
                        counter+=1
                    predicted_spans.append(temp_list)
                    outside_counter+=1
                
                #print("Predicted Spans")
                #print(predicted_spans)
                #print(target_tags)
                evl.run_evaluation(tag_preds= predicted_tags, tag_gold= target_tags , gold_spans=target_nodes, predicted_spans=predicted_spans, label = label_type)
            else:
                predicted_spans = []
                predicted_labels = []
                #print(initial_predicted_list)
                outside_counter = 0
                predicted_tags = []
                for sublist in initial_predicted_list:
                    counter = 0
                    temp_list = []
                    other_temp_list =[]
                    sentence = sentence_list[outside_counter]
                    for token in sublist:
                        temp_tag = token.value
                        predicted_tags.append(temp_tag)
                        other_temp_list.append(temp_tag)
                        tup = (sentence[counter].text, "XX")
                        temp_list.append(tup)
                        counter+=1
                    predicted_spans.append(temp_list)
                    predicted_labels.append(other_temp_list)
                    outside_counter+=1
                gold_target_tags = [*filter(utils.check_space, target_tags)]

                evl.run_evaluation(tag_preds=predicted_tags, tag_gold= gold_target_tags, gold_spans= target_nodes, predicted_spans=[predicted_spans,predicted_labels], label=label_type )

    def hyperparameter_tuning(label_type, model_path, tagtype, epochs, embedding,):
        """ Tuning method for dropout

        Parameters
        ----------
        label_type : String
            Label for model, either surface_ner or linearised_labels
        model_path : String
            Path to model that includes final-model.pt
        tagtype : String
            
        epochs : int
            Number of epochs for tuning
        embedding : String
            Embedding type - either BERT or Glove
        """

       
        # define your search space
        search_space = SearchSpace()
        corpus = DL.create_corpus()

        if embedding == "BERT":
            embeddings = TransformerWordEmbeddings(model = 'SpanBERT/spanbert-base-cased',
                                                fine_tune=True,
                                                subtoken_pooling = 'first_last')
        else:
            embeddings = WordEmbeddings('glove')

        #Embeddings must always be filled, even if just trialing one
        search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[embeddings])
        search_space.add(Parameter.DROPOUT, hp.uniform, low=0.2, high=0.5)

        #only one training run for resource reasons
        param_selector = SequenceTaggerParamSelector(corpus,
                                             tagtype,
                                             model_path,
                                             max_epochs=epochs)
        experiment = False

        if experiment:
            # start the optimization
            print("Optimising the current model.......")
            param_selector.optimize(search_space, max_evals=1)

    def find_opt_learning_rate(self, mod_path, itrs, mini_b_size):
        """Uses cyclical learning rates to find the optimal learning rate for a model

        Parameters
        ----------
        mod_path : String
            Path to model
        itrs : int
            Number of iterations for the learning rate cycle
        mini_b_size : int
            Batch size for model
        """
        print(f"Finding opt learning rate with {itrs}# of iterations and batch size {mini_b_size} ")
        trained_model = SequenceTagger.load(model_path = f"{mod_path}/final-model.pt")
        corpus = DL.create_corpus()
        
        find_learning_rate = False
        if find_learning_rate:
            #initialize trainer
            trainer: ModelTrainer = ModelTrainer(trained_model, corpus)
            #find learning rate
            learning_rate_tsv = trainer.find_learning_rate(mod_path, Adam, end_learning_rate=0.1, iterations=itrs ,mini_batch_size=mini_b_size)

