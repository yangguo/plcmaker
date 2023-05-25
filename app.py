import os
import streamlit as st
from maker import txt2df, df2section, groupbysection, exclude_row,savedf2csv
from utils import (  
    # combine_df_columns,
    get_folder_list,
    get_section_list,
)
from checkrule import searchByName,searchByItem,searchByNamesupa
from gptfuc import (
    add_ruleindex,
    build_ruleindex,
    delete_db,)

rulefolder = "rules"

def main():

    st.header("监管制度转换")

    industry_list = get_folder_list(rulefolder)

    industry_choice = st.sidebar.selectbox("选择行业:", industry_list)

    match = st.sidebar.radio("搜索方式", ("格式转换","生成模型", "模型管理"))

    if match == "格式转换":
        st.subheader("格式转换")

        # get plcname input
        plc_name = st.text_input('输入制度名称')
        plc_text = st.text_area('输入制度内容')

        section_input = st.sidebar.text_input('输入制度段落模版',
                                            value='第\w章 第\w节 第\w??\w?\w条')
        # input exclude text list
        exclude_text = st.sidebar.text_input('输入排除文本',
                                            value='^\d\d?$')

        # checkbox for save to db
        save_checkbox = st.sidebar.checkbox('保存到数据库', value=False)

        if st.sidebar.button('格式转换'):
            # check if plc_name and plc_text are empty
            if plc_text == '' or plc_name == '':
                st.warning('请输入制度名称和内容')
            else:
                # convert text to dataframe
                df = txt2df(plc_text)
                st.subheader('第一步: 输入文本')
                st.table(df)
                # get section list
                section_list = section_input.split()
                # print section list
                # convert df by section list
                df1, sectionls = df2section(df, section_list)
                st.subheader('第二步: 按段落转换')
                st.write(df1)
                # print sectionls
                st.warning('段落列表:')
                st.write(sectionls)

                # get exclude text list
                exclude_list = exclude_text.split()
                df2, exlist = exclude_row(df1, exclude_list)
                st.subheader('第三步: 排除文本转换')
                st.write(df2)
                # print exclude list
                st.warning('排除文本列表:')
                st.write(exlist)

                # group by section
                df3, tiaols = groupbysection(df2, section_list)
                st.subheader('第四步: 按段落分组转换条文')
                st.write(df3)
                # print tiao list
                st.warning('条文列表:')
                st.write(tiaols)

                # print total number
                st.sidebar.write('条文数量:', len(df3))

                # convert and download df as csv
                st.sidebar.download_button(label='下载制度条文',
                                        data=df3.to_csv(),
                                        file_name=plc_name + '.csv',mime='text/csv')
                
                # save df as csv
                # savedf(df3, plc_name)
                # save button
                if save_checkbox:
                    # rename columns
                    df3.rename(columns={"txt": "条款", "section": "结构"}, inplace=True)
                    # add save path under rulefolder
                    save_path = os.path.join(rulefolder, industry_choice)
                    # save df as csv
                    savedf2csv(df3, plc_name, save_path)
                    st.sidebar.success('保存成功')

    elif match == "生成模型":
        st.subheader("生成模型")

        name_text = ""
        searchresult, choicels = searchByName(name_text, industry_choice)

        make_choice = st.sidebar.multiselect("选择监管制度:", choicels)

        if make_choice == []:
            make_choice = choicels
        section_list = get_section_list(searchresult, make_choice)
        column_text = st.sidebar.multiselect("选择章节:", section_list)
        if column_text == []:
            column_text = ""
        else:
            column_text = "|".join(column_text)

        fullresultdf, total = searchByItem(
            searchresult, make_choice, column_text, ""
        )
        # reset index
        fullresultdf = fullresultdf.reset_index(drop=True)
        st.write(fullresultdf)
        # get total number of results
        total = fullresultdf.shape[0]
        st.markdown("共搜索到" + str(total) + "条结果")
        # metadata = fullresultdf[['监管要求','结构']].to_dict(orient="records")
        # st.write(metadata)
        # button to build model
        build_model = st.button("新建模型")
        if build_model:
            with st.spinner("正在生成模型..."):
                build_ruleindex(fullresultdf, industry_choice)
                st.success("模型生成完成")

        add_model = st.button("添加模型")
        if add_model:
            with st.spinner("正在添加模型..."):
                add_ruleindex(fullresultdf, industry_choice)
                st.success("模型添加完成")


    elif match == "模型管理":
        st.subheader("模型管理")

        name_text = ""
        searchresult, choicels = searchByNamesupa(name_text, industry_choice)

        make_choice = st.sidebar.multiselect("选择监管制度:", choicels)

        if make_choice == []:
            make_choice = choicels
        section_list = get_section_list(searchresult, make_choice)
        column_text = st.sidebar.multiselect("选择章节:", section_list)
        if column_text == []:
            column_text = ""
        else:
            column_text = "|".join(column_text)

        fullresultdf, total = searchByItem(
            searchresult, make_choice, column_text, ""
        )
        # reset index
        fullresultdf = fullresultdf.reset_index(drop=True)
        st.write(fullresultdf)
        # get total number of results
        total = fullresultdf.shape[0]
        st.markdown("共搜索到" + str(total) + "条结果")


        # delete model button
        delete_model = st.button("删除模型")
        if delete_model:
            with st.spinner("正在删除模型..."):
                try:
                    delete_db(industry_choice, make_choice)
                    st.success("模型删除成功")
                except Exception as e:
                    st.error(e)
                    st.error("模型删除失败")

if __name__ == '__main__':
    main()