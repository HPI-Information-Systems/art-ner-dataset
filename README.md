# Generation of Training Data for Named Entity Recognition of Artworks
Data and pre-trained models from the paper [Generation of Training Data for Named Entity Recognition of Artworks](http://www.semantic-web-journal.net/content/generation-training-data-named-entity-recognition-artworks) published in the Semantic Web Journal 2023 issue. 

## Data 

Pending approval/license by the owner of the corpus.

## Models

The models can be downloaded from [here](https://figshare.com/articles/software/NER_models/22010696)
<!--- (https://owncloud.hpi.de/s/UbefgKazzkn1uU9) -->

### SpaCy

The Spacy pre-trained model 'en_core_web_md' was used a baseline for further training with domain related annotations. The version of Spacy is 3.3.0. Documentation related to the same is available [here](https://github.com/explosion/spacy-models/releases/tag/en_core_web_md-3.3.0). 

To use the spacy model to annotate a file with texts (see spacy_model/example_file.csv), download the model folder and run the script spacy_model/run_spacy.py as follows

```python run_spacy.py model_location example_file.csv```


### Flair

The Flair model was trained using GloVe (`en-glove`) and forward and backward Flair Embeddings (`news-X`). More information on these embedding models can be found in Flair's [documentation](https://github.com/flairNLP/flair/blob/2fde64610244c7706cef68c882b9ce0e96261d2d/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md)

In order to run the model with a sentence, the script flair_model/RunNER.py can be executed with the following command

```python RunNER.py final-model.pt "This is a sentence"```
