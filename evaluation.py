import deep_deep_eval_edm as evl_edm
import torch
import utils as utils
import tree2spans as ts
from torchmetrics import F1Score
import data_loader as DL
from flair.data import Dictionary

def calculate_F1_score(predicted, target):
    """Calculates tag accuracy

    Parameters
    ----------
    predicted : [Strings]
        List of tags predicted by models
    target : [Strings]
        List of gold tags from the Redwoods Corpus

    Returns
    -------
    _type_
        _description_
    """
    f1_score = F1Score(average = 'micro')
    result = f1_score(predicted, target)
    return result              

def run_evaluation(tag_preds, tag_gold, gold_spans, predicted_spans, label):
    """Calculates precision, recall and F1 score 

    Parameters
    ----------
    tag_preds : [Strings]
        List of predicted tags
    tag_gold : [Strings]
        List of gold tags
    gold_spans : [[Strings]]
        List containing a list of lists
    predicted_spans : [[Strings]]
        List containing a list of lists
    label : String
        Label types - either surface_ner or linearised_labels (abstract)
    """
    print("Tag Accuracy")
    ner_dictionary = Dictionary
    ner_dictionary = ner_dictionary.load_from_file(f"../corpus/corpus_label_dict_{label}")

    pred_tag = torch.tensor(utils.get_numeric_value_for_text_labels(tag_preds, ner_dictionary), dtype= int)

    targ_tag = torch.tensor(utils.get_numeric_value_for_text_labels(tag_gold, ner_dictionary), dtype= int)

    f1 = calculate_F1_score(pred_tag,targ_tag)
    print(f"F1: {f1}")
    
    if label == 'surface_ner':
        node_pred = True
        if node_pred:
            #print("Node Prediction")
            temp = arrange_spans(gold_spans)
            print(temp[0])
            ##gold = format_spans_for_evaluation(arrange_spans(gold_spans))
            
            #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            #predictions = format_spans_for_evaluation(arrange_spans(predictions_list))
            
            #print(len(predicted_spans))
            temp2 = arrange_spans(predicted_spans)
            print(temp2[0])
            ##predictions =  format_spans_for_evaluation(arrange_spans(predicted_spans))
            #print(len(predictions))
            ##evl_edm.perform_evaluation(gold = gold, pred= predictions, inref=None, verb=True)

            #print("Gold spans")
            #print(gold)
            #print("Predicted spans")
            #print(predictions)

    elif label == "linearised_labels":
        gold_spns = ts.sequence_to_parenthesis(sentences=gold_spans[0], labels=gold_spans[1])
        gold_trees = ts.trees_to_spans(gold_spns)
        formatted_gold_spans = format_spans_for_evaluation(gold_trees,True)
        
        pred_spans = ts.sequence_to_parenthesis(sentences=predicted_spans[0],labels = predicted_spans[1])
        predicted_trees = ts.trees_to_spans(pred_spans)
        formatted_predicted_spans = format_spans_for_evaluation(predicted_trees,True)
        evl_edm.perform_evaluation(gold = formatted_gold_spans, pred = formatted_predicted_spans, inref=None, verb = True)

        #print("Gold trees")
        #print(gold_spans[0])
        #print(gold_spans[1])
        #print("Gold trees to spans")
        #gold_trees[0]
        #gold_trees[1]
        
        #print("Predictions - Tags + lables")
        #print(predicted_spans[0][0])
        #print(predicted_spans[0][1])
        #print(predicted_spans[1][0])
        #print(predicted_spans[1][1])
        #print("Predicted trees")
        #print(pred_spans[0])
        #print(pred_spans[1])
        
        #print("Predicted spans to trees")
        #print(predicted_trees[0])
        #print(predicted_trees[1])
        
        print("Gold Spans")
        print(formatted_gold_spans)
        print("Predicted Spans")
        print(formatted_predicted_spans)

def combined_evaluation():
    """Runs full evaluation combining surface and abstract models"""
    abs_gold = utils.abs_gold
    surf_gold = utils.surf_gold

    combined_gold = [ tup[0]+";"+ tup[1] for tup in zip(abs_gold, surf_gold)]
    mapped_combined_gold = [*map(utils.remove_semi_colon,combined_gold)]

    #Abs Glove + Surf Glove + No CRF
    abstract_glove = utils.abs_glove_pred

    surf_glove_nc = utils.surf_glove_nc
    combination_one = [ tup[0]+";"+ tup[1] for tup in zip(abstract_glove, surf_glove_nc)]
    mapped_combination_one = [*map(utils.remove_semi_colon,combination_one)]

    print("Abstract Glove + Surf Glove without CRF")
    evl_edm.compute_f1(gold_set = mapped_combined_gold, predicted_set=mapped_combination_one,inref=None,verbose=True)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    #Abs Glove + Surf Glove + CRF
    surf_glove_c = utils.surf_glove_c
    combination_two = [ tup[0]+";"+ tup[1] for tup in zip(abstract_glove, surf_glove_c)]
    mapped_combination_two = [*map(utils.remove_semi_colon,combination_two)]

    print("Abstract Glove + Surf Glove with CRF")
    evl_edm.compute_f1(gold_set = mapped_combined_gold, predicted_set=mapped_combination_two,inref=None,verbose=True)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    #Abs BERT + Surf BERT + No CRF
    abstract_bert = utils.abs_bert_pred

    surf_bert_nc = utils.surf_bert_nc
    combination_three = [ tup[0]+";"+ tup[1] for tup in zip(abstract_bert, surf_bert_nc)]
    mapped_combination_three = [*map(utils.remove_semi_colon,combination_three)]

    print("Abstract BERT + Surf BERT without CRF")
    evl_edm.compute_f1(gold_set = mapped_combined_gold, predicted_set=mapped_combination_three,inref=None,verbose=True)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    #Abs BERT + Surf BERT + CRF

    surf_bert_c = utils.surf_bert_c
    combination_four = [ tup[0]+";"+ tup[1] for tup in zip(abstract_bert, surf_bert_c)]
    mapped_combination_four = [*map(utils.remove_semi_colon,combination_four)]

    print("Abstract BERT + Surf BERT with CRF")
    evl_edm.compute_f1(gold_set = mapped_combined_gold, predicted_set=mapped_combination_four,inref=None,verbose=True)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

def format_spans_for_evaluation(spans, linear = False):
    """Formats spans in tuple format into format for deep deep eval

    Parameters
    ----------
    spans : [[]]
        List of lists of tuples
    linear : bool, optional
        Flag for if formatting spans for abstract labels, by default False

    Returns
    -------
    [Strings]
        List of formatted strings
    """
    all_formatted_spans = []
    for span in spans:   
        if linear:
            formatted_list = [f"{tup[1]}:{tup[2]} XX {tup[0]};" for tup in span]
        else:
            formatted_list = [f"{tup[0]}:{tup[1]} {tup[3]} {tup[2]};" for tup in span]

        full_list = "".join(formatted_list)
        full_list = full_list.strip(";")
        all_formatted_spans.append(full_list)
    return all_formatted_spans
  
def arrange_spans(span_tuples):
    """Arranges tuples into span tuples

    Parameters
    ----------
    span_tuples : [[()]]
        List of lists of tuples for each sentence

    Returns
    -------
    [[()]]
        Lists of lists of span tuples for a sentence
    """
    unformatted_predictions= []
    for predicted_taglist in span_tuples:
        temp_list = []
        search_for_span = True
        skip = 0
        start = 0
        current = predicted_taglist[0]
        while search_for_span:
            if start < len(predicted_taglist):
                current_tuple = predicted_taglist[start]
                if start == len(predicted_taglist)-1: # we've reached the last tuple and it hasn't been skipped, thus just add
                    temp_list.append(current)
                    search_for_span = False
                    break
                else:
                    for k in range(start+1, len(predicted_taglist)):
                        next_tuple = predicted_taglist[k]
                        final_tuple = current_tuple
                        result = compare_tuples(current_tuple, next_tuple)

                        if result is None:
                            temp_list.append(final_tuple)
                            break
                        else:
                            final_tuple = result
                            current_tuple = result
                            skip+=1
                            if k == (len(predicted_taglist)-1): #comparing last tuple to current and they matched
                                temp_list.append(final_tuple)
                                break
                start = start + skip +1
                skip = 0
            else:
                break
        unformatted_predictions.append(temp_list)
        

    return unformatted_predictions            

def compare_tuples(tup1,tup2):
    """Helper method for arrange_spans - compares to tuples to check if they're part of the same span

    Parameters
    ----------
    tup1 : ()
        Current tuple
    tup2 : ()
        Next tuple

    Returns
    -------
    () or None
        If tuples are part of span, returns new span, otherwise returns None
    """
    tag1 = tup1[2]
    tag2 = tup2[2]
    if tag1 == tag2:
        #print("Span found")
        new_tup = (tup1[0], tup2[0], f"{tup1[3]} {tup2[3]}", tag1)
        #print(new_tup)
        return  new_tup
    else:
        #print("Not Span")
        return None
                