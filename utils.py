import streamlit as st
import torch

from transformers import RoFormerModel, RoFormerTokenizer

modelfolder = 'junnyu/roformer_chinese_sim_char_ft_base'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = RoFormerTokenizer.from_pretrained(modelfolder)
    model = RoFormerModel.from_pretrained(modelfolder)
    model = model.to(device)
    return model, tokenizer

model, tokenizer = load_model()


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