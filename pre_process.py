# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 22:44:26 2023

@author: saina
"""

import string
import nltk
import torch
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.voc_ import vocab

# Load the captions
def load_captions(file_path):
    
    """
    Load the captions from the captions.txt file.
    :param captions_file_path: The path to the captions.txt file.
    :return: A dictionary with image names as keys and lists of captions as values.
    """
    with open(file_path, "r") as f:
        captions = f.read().strip().split("\n")
    captions_dict = {}
    
    # print(caption)
    for caption in captions:
        print(caption)
        img_name, caption_text = caption.split("\t")
        img_name = img_name.split("#")[0]
        if img_name not in captions_dict:
            captions_dict[img_name] = []
        captions_dict[img_name].append(caption_text)
    return captions_dict

# Preprocess the captions
def preprocess_captions(captions_dict):
    
    """
    Preprocess the captions by removing punctuation, converting to lowercase, and tokenizing.
    :param captions_dict: A dictionary with image names as keys and lists of captions as values.
    :return: A dictionary with image names as keys and lists of preprocessed captions as values.
    """
    preprocessed_captions = {}
    for img_name, captions in captions_dict.items():
        preprocessed_captions[img_name] = []
        for caption in captions:
            caption = caption.lower()
            caption = "".join([char for char in caption if char not in string.punctuation])
            preprocessed_captions[img_name].append(caption)
    return preprocessed_captions

# Create a voc_ulary
def create_vocabulary(preprocessed_captions, voc__size=5000):
    """
    Create a voc_ulary from the preprocessed captions.
    :param preprocessed_captions: A dictionary with image names as keys and lists of preprocessed captions as values.
    :return: A voc_ulary object.
    """
    all_captions = " ".join([caption for captions in preprocessed_captions.values() for caption in captions])
    words = nltk.word_tokenize(all_captions)
    word_counts = Counter(words)
    most_common_words = [word for word, count in word_counts.most_common(voc__size - 1)]
    most_common_words.append("<unk>")
    
    print(most_common_words)
    voc__ = vocab(Counter(most_common_words), specials=("<pad>", "<start>", "<end>"))
    return voc__

# Convert captions to sequences
def captions_to_sequences(preprocessed_captions, voc_):
    """
    Convert the preprocessed captions to sequences of integers.
    :param preprocessed_captions: A dictionary with image names as keys and lists of preprocessed captions as values.
    :param voc_: A voc_ulary object.
    :return: A dictionary with image names as keys and lists of caption sequences as values.
    """
    tokenizer = get_tokenizer("basic_english")
    sequences = {}
    for img_name, captions in preprocessed_captions.items():
        sequences[img_name] = []
        for caption in captions:
            words = tokenizer(caption)
            word_indices = [voc_.get_stoi()[word] if word in voc_.get_stoi() else voc_.get_stoi()["<unk>"] for word in words]
            sequences[img_name].append([voc_.get_stoi()["<start>"]] + word_indices + [voc_.get_stoi()["<end>"]])
    return sequences




def load_captions(file_path):
    
    """
    Load the captions from the captions.txt file.
    :param captions_file_path: The path to the captions.txt file.
    :return: A dictionary with image names as keys and lists of captions as values.
    """
    with open(file_path, "r") as f:
        captions = f.read().strip().split("\n")
    captions_dict = {}
    
    # print(caption)
    for caption in captions:
        
        if '.jpg,' not in caption: continue
    
        img_name, caption_text = caption.split(".jpg,")
        img_name+='.jpg'
        img_name = img_name.split("#")[0]
        if img_name not in captions_dict:
            captions_dict[img_name] = []
        captions_dict[img_name].append(caption_text)
    return captions_dict


# Path to your captions.txt file
captions_file_path = ".//Data//captions.txt"

# Load captions from the file
captions_dict = load_captions(captions_file_path)

# Preprocess the captions
preprocessed_captions = preprocess_captions(captions_dict)

# Create a voc_ulary
voc_ = create_vocabulary(preprocessed_captions)

# Convert captions to sequences
sequences = captions_to_sequences(preprocessed_captions, voc_)

print(sequences)




# voc_.get_stoi()['salvar']

# from torchtext.voc_ import voc_
# from collections import Counter, OrderedDict
# counter = Counter(["a", "a", "b", "b", "b"])
# sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
# ordered_dict = OrderedDict(sorted_by_freq_tuples)
# v1 = voc_(ordered_dict)
# print(v1['a']) #prints 1
# print(v1['out of voc_']) #raise RuntimeError since default index is not set

# tokens = ['e', 'd', 'c', 'b', 'a']
# #adding <unk> token and default index
# unk_token = '<unk>'
# default_index = -1
# v2 = voc_(OrderedDict([(token, 1) for token in tokens]), specials=[unk_token])
# v2.set_default_index(default_index)
# print(v2['<unk>']) #prints 0
# print(v2['out of voc_']) #prints -1
# #make default index same as index of unk_token
# v2.set_default_index(v2[unk_token])
# v2['out of voc_'] is v2[unk_token] #prints True