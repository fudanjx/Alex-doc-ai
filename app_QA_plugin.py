
from langchain.tools import DuckDuckGoSearchRun
import pandas as pd
import duckduckgo_search as DDG
import streamlit as st

def search_web(site, user_query):
    search = DuckDuckGoSearchRun()
    if site == 'Search Internet':
        results = search.run(user_query)
    else:
        results = search.run(f"site:{site} {user_query}")
    return results


def retrieve_speciality_plugin():
    plugin_df_dase = pd.read_csv('https://www.dropbox.com/s/tqm8riwx7d8cs59/plugin_template.csv?dl=1')

    plugin_list = plugin_df_dase['plugin_site'].tolist()
    
    option = st.selectbox(
        '##### Select speciality plugin:',
        plugin_list)   
    df_plugin_selection = plugin_df_dase[plugin_df_dase['plugin_site'] == option]
    defult_prompt = df_plugin_selection['prompt'].values[0]
    fix_prompt = df_plugin_selection['fix_prompt'].values[0]
    return option, defult_prompt , fix_prompt

