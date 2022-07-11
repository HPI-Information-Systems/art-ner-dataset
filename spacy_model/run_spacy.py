# from __future__ import unicode_literals, print_function
import random
import glob
import os
import time
import glob
import pandas as pd
import numpy as np
import spacy
import io
import json
from bs4 import BeautifulSoup
import numpy as np
import argparse


spacy.require_gpu()
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def format_input(input_file):
    print(input_file)
    all_contents = pd.read_csv(input_file, encoding="utf-8", header=None)
    contents = all_contents.drop_duplicates()
    contents_list = contents.iloc[:, 0].tolist()
    # print(contents_list)
    # print ("Sentences : ",len(contents_list))
    input_data = []

    data = []
    no_text = 0
    for text in range(len(contents_list)):
        # print (texts[text])
        if (contents_list[text] is np.nan):
            continue

        no_text = no_text + 1
        if (no_text % 100 == 0):
            print()
            print("Iteration: ", no_text)
            print("Size of data: ", len(data))

        entries = contents_list[text].splitlines()
        for entry in entries:
            clean_entry = BeautifulSoup(entry, "lxml").text
            clean_entry = clean_entry.replace('\t','')  
	    # important else \t are shown tabs and there are empty strings in final iob file
            # print (clean_entry)

            ner_entities = []
            entities = {}
            entities["entities"] = ner_entities
            # print(entities)
            entry_data = [clean_entry, entities]
            data.append(entry_data)
            # print (entry_data)

    # add all anno from all files
    # files_anno = files_anno.append(text_anno)

    input_data = [(clean_entry, entities) for clean_entry, entities in data]
    print(len(input_data))

    return input_data


def run_model(model_dir, input_data):
    # run the saved model
    print("Loading from", model_dir)
    #nlp = spacy.load(model_dir)

    #for baseline evaluation
    nlp = spacy.load("en_core_web_md")

    text_anno = []
    for text, _ in input_data:
        # print(text)
        if not text:
            continue
        doc = nlp(text)
        ner_entities = []
        for ent in doc.ents:
            # if (ent.label_ in 'WORK_OF_ART'):
            if ((ent.label_ in 'TITLE') or (ent.label_ in 'WORK_OF_ART')):  # both labels refer to artwork mentions
                # print("Entities", [(ent.text, ent.label_)])
                # print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
                ent_text = ent.text
                label = ent.label_
                label = 'WORK_OF_ART'
                start = ent.start_char
                end = ent.end_char
                ner_entity = (ent_text, start, end, label)
                ner_entities.append(ner_entity)
        text_anno.append([text, ner_entities])

    model_anno = pd.DataFrame(text_anno)
    model_anno.columns = ['model_text', 'model_annotations']

    return model_anno


if __name__ == '__main__': 

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_dir',help='Location of the model',type=str)
    parser.add_argument('input_file',help='File with texts to annotate',type=str)
    
    args = parser.parse_args()   

    input_data = format_input(args.input_file)
    # getting the annotations on the data with Spacy model
    model_anno = run_model(args.model_dir, input_data)   
   
    print(model_anno.head())


