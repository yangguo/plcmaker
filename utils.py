import os, glob
import numpy as np
import pandas as pd
import streamlit as st
import asyncio
import torch

from transformers import RoFormerModel, RoFormerTokenizer

modelfolder = 'junnyu/roformer_chinese_sim_char_ft_base'
rulefolder = 'rules'

tokenizer = RoFormerTokenizer.from_pretrained(modelfolder)
model = RoFormerModel.from_pretrained(modelfolder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)


def roformer_encoder(sentences):
    # Tokenize sentences
    encoded_input = tokenizer(sentences,
                              max_length=512,
                              padding=True,
                              truncation=True,
                              return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(
        model_output, encoded_input['attention_mask']).numpy()
    return sentence_embeddings


@st.cache
def get_csvdf(rulefolder):
    files2 = glob.glob(rulefolder + '**/*.csv', recursive=True)
    dflist = []
    for filepath in files2:
        basename = os.path.basename(filepath)
        filename = os.path.splitext(basename)[0]
        newdf = rule2df(filename, filepath)[['监管要求', '结构', '条款']]
        dflist.append(newdf)
    alldf = pd.concat(dflist, axis=0)
    return alldf


def rule2df(filename, filepath):
    docdf = pd.read_csv(filepath)
    docdf['监管要求'] = filename
    return docdf


def get_embedding(folder, emblist):
    dflist = []
    for file in emblist:
        filepath = os.path.join(folder, file + '.npy')
        embeddings = np.load(filepath)
        dflist.append(embeddings)
    alldf = np.concatenate(dflist)
    return alldf


def get_rulefolder(industry_choice):
    # join folder with industry_choice
    folder = os.path.join(rulefolder, industry_choice)
    return folder
