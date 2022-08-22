from flair.models import SequenceTagger
from flair.data import Sentence


# Model 1 
#  load the model you trained
model_1 = SequenceTagger.load('nodepred_graphparsing/models/Glove+No CRF/final-model.pt')

# create example sentence
sentence = Sentence("When bank financing for the buy-out collapsed last week, so did UAL \'s stock")

# predict tags and print
model_1.predict(sentence)
print(sentence.to_tagged_string())

# Model 2

 