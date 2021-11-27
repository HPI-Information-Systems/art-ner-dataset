from flair.models import SequenceTagger
from flair.data import Sentence
import flair
from flair.tokenization import SpacyTokenizer
from spacy import blank
import argparse
import torch

spacy_tokenizer=SpacyTokenizer(blank('en'))
model=None

def annotate(text):
    sentence = Sentence(text,use_tokenizer=spacy_tokenizer)
    model.predict(sentence)
    for span in sentence.get_spans('ner'):
        print(span)
        label=span.get_labels()[0].value
        ent_text=span.text
        start = span.start_pos
        end = span.end_pos
        print(f'An entity type:{label},text:"{ent_text}",[{start}-{end}]')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model',help='Model to use',type=str)
    parser.add_argument('text',help='Text to annotate',type=str)
    parser.add_argument('-cd','--cuda_device',help='Cuda device number (-1 to force cpu). By default its flair´s default behaviour',type=int)
    args = parser.parse_args()
    cuda_device=args.cuda_device
    if cuda_device is not None:
        if cuda_device==-1:
            flair.device=torch.device("cpu")
        else:
            flair.device=torch.device(f"cuda:{cuda_device}")
    print(args)
    model = SequenceTagger.load(args.model)
    annotate(args.text)
