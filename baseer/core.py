from baseer import models
import clip

import os
import torch
import numpy as np
from PIL import Image
import spacy
import nltk
from translate import Translator

from textblob import TextBlob

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

"""
The Baseer class loads the BLIP and CLIP models, and provides methods to use them.

Methods:

__init__ (self, device='gpu'): Initialize the Baseer class.
blip_predict (self): Predict a caption for an image using the BLIP model.
clip_predict (self, sentences): Predict the most relevant sentence for an image using the CLIP model.
translate (self, text, to_lang='en'): Translate text from one language to another.
"""

class Baseer:

    # Variables
    # img, blip_model, blip_processors, clip_model, clip_preprocess,
    # objects_data_path, objects_list, objects_list_ar, objects_list_en
    # ignore_list_path, ignore_list

    objects_data_path = os.path.join(os.path.dirname(__file__), 'data/objects.txt')

    ignore_list_path = os.path.join(os.path.dirname(__file__), 'data/ignore_list.txt')

    def __init__(self,device='gpu'):
        if (device in ['gpu','cuda','GPU','CUDA']):
            if torch.cuda.is_available():
                self.device = 'gpu'
            else:
                self.device = 'cpu'
                print(f"[!] GPU/CUDA device is not available! Using CPU instead.")
        elif device in ['cpu','CPU']:
            self.device = 'cpu'
            print('[!] Using CPU device by choice.')
        else:
            raise ValueError(f"Invalid device: {device}, please use 'gpu' or 'cpu'.")
        
        self.BLIP_model = models.BLIPModel(self.device)
        self.CLIP_model = models.CLIPModel(self.device)

        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        self.nlp = spacy.load("en_core_web_lg")
        
        self.load_objects_data()
        self.load_ignore_list()

    def set_img(self, img_path):
        img = Image.open(img_path)
        img = img.convert('RGB')
        self.img = img

    def BLIP_predict(self, image: Image = None):
        if image is None:
            image = self.img
        return self.BLIP_model.predict(image)
    
    def CLIP_predict(self, sentences, return_probs=False, image: Image = None):
        if image is None:
            image = self.img
        return self.CLIP_model.predict(image, sentences, return_probs)
    

    def translate(self,text,method = 'translator',to_lang="ar"):
        if (method == 'translator'):
            translator = Translator(to_lang=to_lang)
            return translator.translate(text)
        
        elif (method == 'other_method'):
            return 0
        
        return Exception('Translator {method} is deprecated or doesn\'t exist.')


    def load_objects_data(self,path=None):
        
        if path is None:
            path = self.objects_data_path

        with open(path, "r") as f:
            objects = f.readlines()

        objects = [word.strip() for word in objects] # remove newlines
        self.objects_list = [word.split(",") for word in objects] # split to [Arabic,English]

        # pick the first item in each list
        self.objects_list_ar = [word[0] for word in self.objects_list]
        self.objects_list_en = [word[1] for word in self.objects_list]

    def load_ignore_list(self, path=None):
        if path is None:
            path = self.ignore_list_path
        with open(path, "r") as f:
            ignore_list = f.readlines()
        self.ignore_list = [word.strip() for word in ignore_list]

    def nlp_object_extractor(self,doc,method = 'nltk_nouns'):
        if (method == 'nltk_nouns'):
            doc = self.nlp(doc)
            return [token.lemma_ for token in doc if token.pos_ == "NOUN"]
        elif (method == 'nltk_full_phrase'): # extract full phrase nouns
            doc = self.nlp(doc)
            return [chunk.text for chunk in doc.noun_chunks]


    def magic_a(self,similar_num=5,verbose=True):

        original_caption = self.BLIP_model.predict(self.img)

        best_caption = original_caption

        nouns = self.nlp_object_extractor(best_caption,'nltk_full_phrase')

        if verbose:
            print("Generating nouns and looping:")
        for noun in nouns:
            if noun in self.ignore_list:
                continue
            if verbose:
                print("\n\n")
                print(f'————————— Noun: {noun} ')

            # top_n_similar_nouns compares the (Extracted English noun) with the English objects in the list and returns the list of [Arabic,English] terms
            top_n_similar_nouns = [(w[0], self.nlp(noun).similarity(self.nlp(w[1])) * 100) for w in self.objects_list]

            if verbose >= 2:
                [ print((w[0], self.nlp(noun).similarity(self.nlp(w[1])) * 100)) for w in self.objects_list ]

            top_n_similar_nouns.sort(key=lambda x: x[1], reverse=True)

            # take the top n similar words
            top_n_similar_nouns = top_n_similar_nouns[:similar_num]
            if verbose:
                print(f'—— Top {similar_num} similar words:')
                [print(term[0],end=" ") for term in top_n_similar_nouns]
                print(noun)
            
            # inject Arabic words
            if verbose:
                print(f'—— Caption:')
                print(best_caption.replace(noun, "{noun}"))
            test_captions = [best_caption]
            for term in top_n_similar_nouns:
                test_captions.append(best_caption.replace(noun, term[0]))
                
            # compare using clip
            if verbose:
                print("—— CLIP Comparison:")
            clip_results = self.CLIP_model.predict(self.img, test_captions, True)
            
            # boosting english term
            clip_results[0][0] = clip_results[0][0] + 0.3
            if verbose:
                print(np.array(clip_results), end='\n\n')


            # the best option in iteration
            best_caption = test_captions[np.argmax(clip_results)]
            if verbose:
                print(f'—— Updated the caption to:\n{best_caption}')

        # translate to Arabic
        if verbose:
            print(f'—— Translating to Arabic:')
            print(best_caption)
        best_caption = self.translate(best_caption)
        if verbose:
            print(best_caption)

        return best_caption
    
    def predict(self,method='magic_a',verbose=True):
        if verbose:
            print(f"Predicting Using ({method})...")
        prediction_method = getattr(self, method)
        return prediction_method(verbose=verbose)
