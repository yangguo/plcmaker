import os
import pandas as pd
import streamlit as st
import numpy as np

from utils import get_csvdf, get_embedding,roformer_encoder

import glob

uploadfolder = 'uploads'

# convert df by section list
def df2section(df, sectionlist):
    # exclude last section
    subsectionls=sectionlist[:-1]
    for i,section in enumerate(subsectionls):
        # section pattern
        section_pattern = '^(第\w?\w?\w'+section+')(.*)'
        # section no and name
        section_no = 'section'+str(i)+'_no'
        section_name = 'section'+str(i)+'_name'
        df[section_no]=df['item'].str.extract(section_pattern)[0]
        df[section_name]=df['item'].str.extract(section_pattern)[1]
    
    # st.write(df)
    # reverse order of subsectionls
    revsubsectionls=subsectionls[::-1]
    # st.write(revsubsectionls)
    # exclude last item
    revsubsectionls=revsubsectionls[:-1]
    # st.write(revsubsectionls)
   # for in reverse range order
    for i in range(len(revsubsectionls)):
        # st.write(i)
        # section no and name
        section_no = 'section'+str(i)+'_no'
        section_name = 'section'+str(i)+'_name'
        # next section no and name
        next_section_no = 'section'+str(i+1)+'_no'
        next_section_name = 'section'+str(i+1)+'_name'
        # fillna with last section no and name
        df[next_section_no].fillna(df[section_no], inplace=True)
        df[next_section_name].fillna(df[section_name], inplace=True)
    # st.write(df)
    # fillna with all subsection no and name
    for i,section in enumerate(subsectionls):
        section_no = 'section'+str(i)+'_no'
        section_name = 'section'+str(i)+'_name'
        df[section_no].fillna(method='ffill',inplace=True)
        df[section_name].fillna(method='ffill',inplace=True)
    # st.write(df)
    # last section
    section_pattern = '^(第\w?\w?\w'+sectionlist[-1]+')(.*)'
    df['tiao']=df['item'].str.extract(section_pattern)[0]
    df['txt']=df['item'].str.extract(section_pattern)[1]
    df['txt'].fillna(df['item'],inplace=True)
    df['tiao'].fillna(method='ffill',inplace=True)
    # st.write(df)
    # exclude row with section name
    for i,section in enumerate(subsectionls):
        section_pattern = '^第\w?\w?\w'+section
        df=df[~df['txt'].str.contains(section_pattern)]
    # st.write(df)
    return df


# group by section
def groupbysection(df, sectionlist):
    # exclude last section
    subsectionls=sectionlist[:-1]
    groupls=[]
    for section in range(len(subsectionls)):
        # section no and name
        section_no = 'section'+str(section)+'_no'
        section_name = 'section'+str(section)+'_name'
        groupls.append(section_no)
        groupls.append(section_name)
    groupls.append('tiao')
    dfout=df.groupby(groupls,sort=False)['txt'].apply(lambda x: ''.join(x)).reset_index()
    # initalize section value
    dfout['section']=''
    # combine all section no and name to one column
    for i,section in enumerate(subsectionls):
        section_no = 'section'+str(i)+'_no'
        section_name = 'section'+str(i)+'_name'
        dfout['section']=dfout['section']+dfout[section_no]+' '+dfout[section_name]+'/'
    dfout['section']=dfout['section']+dfout['tiao']
    return dfout    


# exclude row by exclude list
def exclude_row(df, exclude_list):
    for exclude in exclude_list:
        df=df[~df['item'].str.contains(exclude)]
    return df


def savedf(txtlist, filename):
    df = pd.DataFrame(txtlist)
    df.columns = ['item']
    # print(df)
    # df['制度'] = filename
    # df['结构'] = df.index
    basename = filename.split('.')[0]
    savename = basename + '.csv'
    savepath = os.path.join(uploadfolder, savename)
    # df.to_csv(savepath)
    return df


def txt2df(filename, text):
    itemlist = text.replace(' ', '').split('\n')
    dflist = [item for item in itemlist if len(item) > 0]
    # print(dflist)
    df=savedf(dflist, filename)
    return df   


def get_txtdf():
    fileslist = glob.glob(uploadfolder + '**/*.txt', recursive=True)
    # get csv file name list
    csvfiles = get_csvfilelist()
    for filepath in fileslist:
        # get file name
        name = getfilename(filepath)
        if name not in csvfiles:
            txt2df(name, filepath)


# return corpus_embeddings
def getfilename(file):
    filename = os.path.basename(file)
    name = filename.split('.')[0]
    return name

def sent2emb(sents):
    embls=[]
    count=0
    for sent in sents:
        sentence_embedding=roformer_encoder(sent)
        embls.append(sentence_embedding)    
        # print(count)
        count+=1
    all_embeddings=np.concatenate(embls)
    return all_embeddings

def file2embedding(file):
    df=pd.read_csv(file)
    sentences=df['item'].tolist()
#     sentence_embeddings=roformer_encoder(sentences)
    all_embeddings=sent2emb(sentences)
    name=getfilename(file)
    savename=name+'.npy'
#     all_embeddings = np.array(sentence_embeddings)    
    np.save(savename, all_embeddings)


def encode_plclist():
    files = glob.glob(uploadfolder + '**/*.csv', recursive=True)
    # get npy file name list
    npyfiles = get_npyfilelist()
    for file in files:
        # get file name
        name = getfilename(file)
        # check if file is not in npy file list
        if name not in npyfiles:
            try:
                file2embedding(file)
            except Exception as e:
                st.error(str(e))


# get npy file name list
def get_npyfilelist():
    files2 = glob.glob(uploadfolder + '**/*.npy', recursive=True)
    filelist = []
    for file in files2:
        name = getfilename(file)
        filelist.append(name)
    return filelist


# get csv file name list
def get_csvfilelist():
    files2 = glob.glob(uploadfolder + '**/*.csv', recursive=True)
    filelist = []
    for file in files2:
        name = getfilename(file)
        filelist.append(name)
    return filelist


def get_upload_data(plc_list):
    plcdf = get_csvdf(uploadfolder)
    selectdf = plcdf[plcdf['监管要求'].isin(plc_list)]
    emblist = selectdf['监管要求'].unique().tolist()
    plc_encode = get_embedding(uploadfolder, emblist)
    return selectdf, plc_encode


def save_uploadedfile(uploadedfile):
    with open(os.path.join(uploadfolder, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("上传文件:{} 成功。".format(uploadedfile.name))


def get_uploadfiles():
    fileslist = glob.glob(uploadfolder + '/*.csv', recursive=True)
    filenamels = []
    for filepath in fileslist:
        filename = os.path.basename(filepath)
        name = filename.split('.')[0]
        filenamels.append(name)
    return filenamels


def remove_uploadfiles():
    files = glob.glob(uploadfolder + '/*.*', recursive=True)

    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            st.error("Error: %s : %s" % (f, e.strerror))
