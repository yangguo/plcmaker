import glob
import os

import pandas as pd

# import torch

# from transformers import RoFormerModel, RoFormerTokenizer

# modelfolder = 'junnyu/roformer_chinese_sim_char_ft_base'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rulefolder = "rules"

# load model and tokenizer
# @st.cache(allow_output_mutation=True)
# def load_model():
#     tokenizer = RoFormerTokenizer.from_pretrained(modelfolder)
#     model = RoFormerModel.from_pretrained(modelfolder)
#     model = model.to(device)
#     return model, tokenizer

# model, tokenizer = load_model()


# get folder name list from path
def get_folder_list(path):
    folder_list = [
        folder
        for folder in os.listdir(path)
        if os.path.isdir(os.path.join(path, folder))
    ]
    return folder_list


# get section list from df
def get_section_list(searchresult, make_choice):
    """
    get section list from df

    args: searchresult, make_choice
    return: section_list
    """
    df = searchresult[(searchresult["监管要求"].isin(make_choice))]
    conls = df["结构"].drop_duplicates().tolist()
    unils = []
    # print(conls)
    for con in conls:
        itemls = con.split("/")
        #     print(itemls[:-1])
        for item in itemls[:2]:
            unils.append(item)
    # drop duplicates and keep list order
    section_list = list(dict.fromkeys(unils))
    return section_list


# split string by space into words, add brackets before and after words, combine into text
def split_words(text):
    words = text.split()
    words = ["(?=.*" + word + ")" for word in words]
    new = "".join(words)
    return new


def get_rulefolder(industry_choice):
    # join folder with industry_choice
    folder = os.path.join(rulefolder, industry_choice)
    return folder


def get_csvdf(rulefolder):
    files2 = glob.glob(rulefolder + "**/*.csv", recursive=True)
    dflist = []
    for filepath in files2:
        basename = os.path.basename(filepath)
        filename = os.path.splitext(basename)[0]
        newdf = rule2df(filename, filepath)[["监管要求", "结构", "条款"]]
        dflist.append(newdf)
    alldf = pd.concat(dflist, axis=0)
    return alldf


def rule2df(filename, filepath):
    docdf = pd.read_csv(filepath)
    docdf["监管要求"] = filename
    return docdf


# def roformer_encoder(sentences):
#     # Tokenize sentences
#     encoded_input = tokenizer(sentences,
#                               max_length=512,
#                               padding=True,
#                               truncation=True,
#                               return_tensors='pt')

#     # Compute token embeddings
#     with torch.no_grad():
#         model_output = model(**encoded_input)

#     # Perform pooling. In this case, max pooling.
#     sentence_embeddings = mean_pooling(
#         model_output, encoded_input['attention_mask']).numpy()
#     return sentence_embeddings


# # Mean Pooling - Take attention mask into account for correct averaging
# def mean_pooling(model_output, attention_mask):
#     # First element of model_output contains all token embeddings
#     token_embeddings = model_output[0]
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(
#         token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
#         input_mask_expanded.sum(1), min=1e-9)
