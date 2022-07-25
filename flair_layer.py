from cProfile import run
from flair.embeddings import WordEmbeddings, StackedEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
import data_loader as DL
import sys
import torch
import gc


def run_flair_tagger(model_number, visualise):

    label_type = 'label'
    corpus = DL.create_corpus()

    # create label dictionary for a NER tagging task
    ner_dictionary = corpus.make_label_dictionary(label_type=label_type)


    # initialise embeddings
    # maybe change this to not be stacked??
    embedding_types_baseline = [

        WordEmbeddings('glove'),
    ]


    embeddings_baseline = StackedEmbeddings(embeddings=embedding_types_baseline)


    # BERT Models 

    # change to SPANbert base cased 
    # SpanBERT/spanbert-large-cased
    embedding_types_bert = [ 

        TransformerWordEmbeddings('bert-base-uncased'),
    ]

    embeddings_bert = StackedEmbeddings(embeddings=embedding_types_baseline)

    if model_number == "1":
        print("Model #1: Glove BiLSTM without CRF")
        # sequence tagger
        tagger_glove_no_crf = SequenceTagger(hidden_size=256,
                                embeddings=embeddings_baseline,
                                tag_dictionary=ner_dictionary,
                                tag_type=label_type,
                                use_crf=False)


        # trainer
        trainer_glove_no_crf = ModelTrainer(tagger_glove_no_crf, corpus)

        # training
        torch.cuda.empty_cache()
        trainer_glove_no_crf.train('resources/taggers/glove_no_crf',
                    learning_rate=0.1,
                    mini_batch_size=10,
                    max_epochs= 10, 
                    checkpoint=True,
                    write_weights=True)

        #visualising results
        if visualise:


            plotter = Plotter()
            plotter.plot_training_curves('loss.tsv')
            plotter.plot_weights('weights.txt')

    elif model_number == "2":
        corpus = corpus.downsample(0.2)
        print("Model #1: Glove BiLSTM with CRF")
        tagger_glove_crf = SequenceTagger(hidden_size=128,
                                embeddings=embeddings_baseline,
                                tag_dictionary=ner_dictionary,
                                tag_type=label_type,
                                use_crf=True)


        trainer_glove_crf = ModelTrainer(tagger_glove_crf , corpus)

        torch.cuda.empty_cache()
        trainer_glove_crf.train('resources/taggers/glove_crf',
                    learning_rate=0.1,
                    mini_batch_size=10,
                    max_epochs= 10)


    ####################################################################



    elif model_number == "3":
        print("Model #1: BERT BiLSTM without CRF")
        # sequence tagger
        tagger_bert_no_crf = SequenceTagger(hidden_size=256,
                                embeddings=embeddings_bert,
                                tag_dictionary=ner_dictionary,
                                tag_type=label_type,
                                use_crf=False)


        # trainer
        trainer_bert_no_crf = ModelTrainer(tagger_bert_no_crf, corpus)

        # training
        torch.cuda.empty_cache()
        trainer_bert_no_crf.train('resources/taggers/bert_no_crf',
                    learning_rate=0.1,
                    mini_batch_size=10,
                    max_epochs= 10)

    else:
        print("Model #1: BERT BiLSTM with CRF")
        tagger_bert_crf = SequenceTagger(hidden_size=256,
                                embeddings=embeddings_bert,
                                tag_dictionary=ner_dictionary,
                                tag_type=label_type,
                                use_crf=True)


        trainer_bert_crf = ModelTrainer(tagger_bert_crf , corpus)

        torch.cuda.empty_cache()
        trainer_bert_crf.train('resources/taggers/bert_crf',
                    learning_rate=0.1,
                    mini_batch_size=20,
                    max_epochs= 10)

def main():
    model = sys.argv[1]
    visualise = True if sys.argv[2] == "1" else False
    print(visualise)
    #print(model)
    #run_flair_tagger(model)

if __name__ == "__main__":
    main()
