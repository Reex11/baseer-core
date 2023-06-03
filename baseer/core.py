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

    objects_data_path = os.path.join(os.path.dirname(__file__), 'data/objects.txt')

    def __init__(self,device='gpu'):
        if (device == 'gpu'):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device(device)
        
        self.BLIP_model = models.BLIPModel()
        self.CLIP_model = models.CLIPModel()

        # self.blip_model, self.blip_processors = models.BLIPModel().load()
        # self.clip_model, self.clip_preprocess = models.CLIPModel().load()
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        self.nlp = spacy.load("en_core_web_lg")
        
        self.load_objects_data()

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

    # def blip_predict(self): # predicts caption
    #     image = self.blip_processors["eval"](self.img).unsqueeze(0).to(self.device)
    #     # generate caption
    #     result = self.blip_model.generate({"image": image})
    #     return result[0]

    # def clip_predict(self, sentences, return_probs=False): # predicts most relevant sentence
    #     image = self.clip_preprocess(self.img).unsqueeze(0).to(self.device)
    #     text = clip.tokenize(sentences).to(self.device)

    #     with torch.no_grad():
    #         image_features = self.clip_model.encode_image(image)
    #         text_features = self.clip_model.encode_text(text)

    #         logits_per_image, logits_per_text = self.clip_model(image, text)
    #         probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
    #     if return_probs:
    #         return probs
    #     else:
    #         return sentences[probs.argmax()]

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

    def nlp_object_extractor(self,doc,method = 'nltk_nouns'):
        if (method == 'nltk_nouns'):
            doc = self.nlp(doc)
            return [token.lemma_ for token in doc if token.pos_ == "NOUN"]
        elif (method == 'nltk_full_phrase'): # extract full phrase nouns
            doc = self.nlp(doc)
            return [chunk.text for chunk in doc.noun_chunks]
        
    def magic_a(self,similar_num=5):

        original_caption = self.BLIP_model.predict(self.img)
        best_caption = original_caption
        
        nouns = self.nlp_object_extractor(original_caption,'nltk_full_phrase')

        print("Generating nouns and looping:")
        for noun in nouns:
            print("\n\n")
            print(f'————————— Noun: {noun} ')

            # top_n_similar_nouns compares the (Extracted English noun) with the English objects in the list and returns the list of [Arabic,English] terms
            top_n_similar_nouns = [(w[0], self.nlp(noun).similarity(self.nlp(w[1])) * 100) for w in self.objects_list]
            top_n_similar_nouns.sort(key=lambda x: x[1], reverse=True)

            # take the top n similar words
            top_n_similar_nouns = top_n_similar_nouns[:similar_num]
            print(f'—— Top {similar_num} similar words:')
            [print(term[0],end=" ") for term in top_n_similar_nouns]
            print()
            
            # inject Arabic words
            print(f'—— Caption:')
            test_captions = [best_caption]
            print(best_caption.replace(noun, "{noun}"))
            for term in top_n_similar_nouns:
                test_captions.append(best_caption.replace(noun, term[0]))
                # print(test_captions[-1])

            # compare using clip
            print("—— CLIP Comparison:")
            clip_results = self.CLIP_model.predict(self.img, test_captions, True)
            print(np.array(clip_results), end='\n\n')

            # the best option in iteration
            best_caption = test_captions[np.argmax(clip_results)]
            print(f'—— Updated the caption to:\n{best_caption}')

        # translate to Arabic
        print(f'—— Translating to Arabic:')
        print(best_caption)
        best_caption = self.translate(best_caption)
        print(best_caption)

        return best_caption
    
    def predict(self,method='magic_a'):
        print(f"Predicting Using ({method})...")
        prediction_method = getattr(self, method)
        return prediction_method()
