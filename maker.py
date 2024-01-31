import os

import numpy as np
import pandas as pd
import streamlit as st

# from utils import roformer_encoder


uploadfolder = "uploads"


# convert df by section list
def df2section(df, sectionlist):
    # exclude last section
    subsectionls = sectionlist[:-1]
    for i, section in enumerate(subsectionls):
        # section pattern
        section_pattern = "^(" + section + ")(.*)"
        # section_pattern = section + '(.*)'
        # section no and name
        section_no = "section" + str(i) + "_no"
        section_name = "section" + str(i) + "_name"
        df[section_no] = df["item"].str.extract(section_pattern)[0]
        df[section_name] = df["item"].str.extract(section_pattern)[1]

    st.write("df0", df)

    # reverse order of subsectionls
    revsubsectionls = subsectionls[::-1]
    # exclude last item
    revsubsectionls = revsubsectionls[:-1]
    # for in reverse range order
    for i in range(len(revsubsectionls)):
        # section no and name
        section_no = "section" + str(i) + "_no"
        section_name = "section" + str(i) + "_name"
        # next section no and name
        next_section_no = "section" + str(i + 1) + "_no"
        next_section_name = "section" + str(i + 1) + "_name"
        # fillna with last section no and name
        df[next_section_no].fillna(df[section_no], inplace=True)
        df[next_section_name].fillna(df[section_name], inplace=True)

    st.write("df1", df)

    # fillna with all subsection no and name
    for i, section in enumerate(subsectionls):
        section_no = "section" + str(i) + "_no"
        section_name = "section" + str(i) + "_name"
        df[section_no].fillna(method="ffill", inplace=True)
        df[section_name].fillna(method="ffill", inplace=True)

    st.write("df2", df)

    # last section
    section_pattern = "^(" + sectionlist[-1] + ")(.*)"
    df["tiao"] = df["item"].str.extract(section_pattern)[0]
    df["txt"] = df["item"].str.extract(section_pattern)[1]
    df["txt"].fillna(df["item"], inplace=True)
    df["tiao"].fillna(method="ffill", inplace=True)

    st.write("df3", df)

    # initialize section value list
    sectionls = dict()
    # exclude row with section name
    for i, section in enumerate(subsectionls):
        section_pattern = "^" + section
        sectionls[section] = df[df["txt"].str.contains(section_pattern)]["txt"].tolist()
        # df=df[~df['txt'].str.contains(section_pattern)]

    for i, section in enumerate(sectionlist):
        section_pattern = section
        # sectionls[section]=df[df['txt'].str.contains(section_pattern)]['txt'].tolist()
        st.write("matched", df[df["txt"].str.contains(section_pattern)])
        df = df[~df["txt"].str.contains(section_pattern)]

    return df, sectionls


# group by section
def groupbysection(df, sectionlist):
    # exclude last section
    subsectionls = sectionlist[:-1]
    groupls = []
    for section in range(len(subsectionls)):
        # section no and name
        section_no = "section" + str(section) + "_no"
        section_name = "section" + str(section) + "_name"
        groupls.append(section_no)
        groupls.append(section_name)
    groupls.append("tiao")
    dfout = (
        df.groupby(groupls, sort=False)["txt"].apply(lambda x: "".join(x)).reset_index()
    )
    # initalize section value
    dfout["section"] = ""
    # combine all section no and name to one column
    for i, section in enumerate(subsectionls):
        section_no = "section" + str(i) + "_no"
        section_name = "section" + str(i) + "_name"
        dfout["section"] = (
            dfout["section"] + dfout[section_no] + " " + dfout[section_name] + "/"
        )
    dfout["section"] = dfout["section"] + dfout["tiao"]
    # get tiao list
    tiaolist = dfout["tiao"].tolist()
    return dfout, tiaolist


# exclude row by exclude list
def exclude_row(df, exclude_list):
    # exclude row by exclude list
    exlist = dict()
    for exclude in exclude_list:
        # get contain exclude item list
        exlist[exclude] = df[df["item"].str.contains(exclude)]["item"].tolist()
        df = df[~df["item"].str.contains(exclude)]
    return df, exlist


def savedf(df, filename):
    basename = filename.split(".")[0]
    savename = basename + ".csv"
    savepath = os.path.join(uploadfolder, savename)
    df.to_csv(savepath)


def txt2df(text):
    # itemlist = text.replace(' ', '').replace('\u3000', '').replace('\u2002','').split('\n')
    itemlist = text.split("\n")
    dflist = [item for item in itemlist if len(item) > 0]
    df = pd.DataFrame(dflist)
    df.columns = ["item"]
    return df


# return corpus_embeddings
def getfilename(file):
    filename = os.path.basename(file)
    name = filename.split(".")[0]
    return name


def savedf2csv(df, filename, folder):
    basename = filename.split(".")[0]
    savename = basename + ".csv"
    # make folder if not exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    savepath = os.path.join(folder, savename)
    df.to_csv(savepath)


# def sent2emb(sents):
#     embls=[]
#     for sent in sents:
#         sentence_embedding=roformer_encoder(sent)
#         embls.append(sentence_embedding)
#     all_embeddings=np.concatenate(embls)
#     return all_embeddings


# def df2embedding(df):
#     sentences=df['条款'].tolist()
#     all_embeddings=sent2emb(sentences)
#     return all_embeddings


# save embedding
def saveembedding(embeddings, name, folder):
    savename = name + ".npy"
    savepath = os.path.join(folder, savename)
    np.save(savepath, embeddings)
