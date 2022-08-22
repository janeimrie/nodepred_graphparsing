from webbrowser import get
from cairo import Surface
import torch
from torchmetrics import F1Score, Recall, Precision
from sklearn.metrics import precision_score, recall_score
import data_loader as DL
from flair.data import Dictionary

def calculate_F1_score(predicted, target):
    f1_score = F1Score(average = 'micro')
    result = f1_score(predicted, target)
    return result

def calculate_precision(predicted, target):
    #precision = Precision(average='micro')
    #result = precision(predicted, target)
    result = precision_score(y_pred=predicted, y_true=target, average='micro')
    return result

def calculate_recall(predicted, target):
    #recall = Recall(average='micro')
    #result = recall(predicted, target)
    result = recall_score(y_pred=predicted, y_true=target, average='micro')
    return result

def run_evaluation(pred, targ, label_type):
    ner_dictionary = Dictionary
    ner_dictionary = ner_dictionary.load_from_file(f"./corpus/corpus_label_dict_{label_type}")
    #predicted_ = torch.tensor(get_numeric_value_for_text_labels(pred, ner_dictionary)))

    #f1 = calculate_F1_score(predicted=predicted_, target = target)
    pred = get_numeric_value_for_text_labels(pred, ner_dictionary) 
    targ = get_numeric_value_for_text_labels(targ, ner_dictionary)
    print(pred)
    prec = calculate_precision(predicted=pred, target = targ)
    rec = calculate_recall(predicted=pred, target = targ)

    return [prec, rec] #[prec]#, rec, f1]

def get_numeric_value_for_text_labels(label_list, ner_dictionary : Dictionary):
    result = ner_dictionary.get_idx_for_items(label_list)
    return result

def make_none_string(label):
    if label == 'None':
        print("here")
        return "---"
    else:
        return label


#python .\nodepred_graphparsing\test.py "C:\Users\LENOVO\Documents\Honours\Proj\Code\nodepred_graphparsing\models\Surface Tokens + Glove+ No CRF\final-model.pt" 0 -e -nt