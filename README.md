# Generation of Training Data for Named Entity Recognition of Artworks
Data and pre-trained models from the paper "Generation of Training Data for Named Entity Recognition of Artworks"

## Data 

Pending approval/license by the owner of the corpus.

## Models

The models can be downloaded from [here](https://owncloud.hpi.de/s/UbefgKazzkn1uU9)

### SpaCy

The Spacy pre-trained model 'en_core_web_md' was used a baseline for further training with domain related annotations. The version of Spacy is 3.3.0. Documentation related to the same is available [here](https://github.com/explosion/spacy-models/releases/tag/en_core_web_md-3.3.0). 

### Flair

The Flair model was trained using GloVe (`en-glove`) and forward and backward Flair Embeddings (`news-X`). More information on these embedding models can be found in Flair's [documentation](https://github.com/flairNLP/flair/blob/2fde64610244c7706cef68c882b9ce0e96261d2d/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md)

In order to run the model with a sentence, the script flair_model/RunNER.py can be executed with the following command

```python RunNER.py final-model.pt "This is a sentence"```
