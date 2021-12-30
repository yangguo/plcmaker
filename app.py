import streamlit as st
from upload import txt2df,df2section,groupbysection,exclude_row

def main():

    st.subheader("Policy text convertor")

    # get plcname input

    plc_name=st.text_input('Input Name')
    plc_text = st.text_area('Input Text')
    section_input=st.sidebar.text_input('Input Section List',value='章 节 条')
    # input exclude text list
    exclude_text = st.sidebar.text_input('Input Exclude Text',value='^\d\d?$')
  
    if st.sidebar.button('Convert'):
        if plc_text != '':
            # print input
            # st.write(plc_name)
            # st.write(plc_text)
            # convert text to dataframe
            df=txt2df(plc_name,plc_text)
            st.subheader('Input Text')
            st.table(df)
            # get section list
            section_list=section_input.split()
            # print section list
            # st.write(section_list)
            # convert df by section list
            df1=df2section(df,section_list)
            st.subheader('Converted Dataframe')
            st.write(df1)
            # get exclude text list
            exclude_list=exclude_text.split()
            df2=exclude_row(df1,exclude_list)
            st.subheader('Excluded Dataframe')
            st.write(df2)

            # group by section
            df3=groupbysection(df2,section_list)
            st.subheader('Grouped Dataframe')
            st.write(df3)

            # convert and download df as csv
            st.sidebar.download_button(label='Download',data=df3.to_csv(),file_name=plc_name+'.csv')
            # st.write(df2.to_csv())
             
                                
if __name__ == '__main__':
    main()