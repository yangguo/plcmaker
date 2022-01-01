import streamlit as st
from maker import txt2df, df2section, groupbysection, exclude_row, df2embedding,savedf,saveembedding


def main():

    st.header("Policy text convertor")

    # get plcname input
    plc_name = st.text_input('Input Name')
    plc_text = st.text_area('Input Text')

    section_input = st.sidebar.text_input('Input Section Pattern List',
                                          value='第\w章 第\w节 第\w?\w?\w条')
    # input exclude text list
    exclude_text = st.sidebar.text_input('Input Exclude Text Pattern List',
                                         value='^\d\d?$')

    if st.sidebar.button('Convert'):
        # check if plc_name and plc_text are empty
        if plc_text != '' and plc_name != '':
            # convert text to dataframe
            df = txt2df(plc_text)
            st.subheader('Step 1: Input Text')
            st.table(df)
            # get section list
            section_list = section_input.split()
            # print section list
            # convert df by section list
            df1, sectionls = df2section(df, section_list)
            st.subheader('Step 2: Converted Dataframe')
            st.write(df1)
            # print sectionls
            st.warning('Section List')
            st.write(sectionls)

            # get exclude text list
            exclude_list = exclude_text.split()
            df2, exlist = exclude_row(df1, exclude_list)
            st.subheader('Step 3: Excluded Dataframe')
            st.write(df2)
            # print exclude list
            st.warning('Excluded item list:')
            st.write(exlist)

            # group by section
            df3, tiaols = groupbysection(df2, section_list)
            st.subheader('Step 4: Grouped Dataframe')
            st.write(df3)
            # print tiao list
            st.warning('Item List:')
            st.write(tiaols)

            # print total number
            st.sidebar.write('Total Item Number:', len(df3))

            # convert and download df as csv
            st.sidebar.download_button(label='Download Table',
                                       data=df3.to_csv(),
                                       file_name=plc_name + '.csv',mime='text/csv')
            
            # save df as csv
            savedf(df3, plc_name)
            
            with st.spinner('Embedding...'):
                embeddings=df2embedding(df3)
                st.sidebar.success('Embedding Done!')

                # download embedding
                st.sidebar.download_button(label='Download Embeddings',data=embeddings.tobytes(),file_name=plc_name+'_embedding.npy',mime='application/octet-stream')

                # save embedding
                saveembedding(embeddings,plc_name)

if __name__ == '__main__':
    main()