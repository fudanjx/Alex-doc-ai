
import anthropic
import pandas as pd
import os
import streamlit as st


def open_file(filepath):
    with open(filepath, 'r') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w') as outfile:
        outfile.write(content)

def upload_patient_notes ():
    # Upload Patient Notes  
    notes_file = st.file_uploader("Upload Anonymized Patient Notes", type=["csv"])
    if not notes_file:
        results_df = 0   
    else:
        notes_df = pd.read_csv(notes_file)
        results_df = notes_df 
    return results_df

def claude(instruct):
    client = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])
    response = client.completion(
        prompt=instruct,
        # stop_sequences = [anthropic.HUMAN_PROMPT],
        model="claude-v1-100k",
        max_tokens_to_sample=20000,
        temperature=0
    )
    text = response['completion']
    return text

def _generate_prompt(anonymized_notes):
    prompt = f"""{anthropic.HUMAN_PROMPT}: You are a medical officer
    {anthropic.HUMAN_PROMPT}: here we have patient's medical notes and inpatient documentation {anonymized_notes} 
    {anthropic.HUMAN_PROMPT}: Please help to analyze the patient's medical records and inpatient documentation to answer below questions 
    {anthropic.HUMAN_PROMPT}: Summarize briefly what are the main reason stop patient from discharge, and categorize into:
    {anthropic.HUMAN_PROMPT}: 1. Whether patient is medically fit to discharge. 2. Caregiver issue 3. Financial related issue 4. other issue 5. patient ready to go home
        \n\n{anthropic.AI_PROMPT}:\n\n"""
    return prompt


def _generate_summary_prompt(anonymized_notes_agg):
    prompt_summary = f"""{anthropic.HUMAN_PROMPT}: You are a medical officer
    {anthropic.HUMAN_PROMPT}: here we have a group of patient's inpatient documentation {anonymized_notes_agg} 
    {anthropic.HUMAN_PROMPT}: Based on each patient's info, summarize the top 5 most common issues which prevent patient from  \
                              discharge, and give example or elaboration. To present the result, provide the statement  \
                              point by point, Rank by importance with percentage of the problem prevalence
            \n\n{anthropic.AI_PROMPT}:\n\n"""
    return prompt_summary
    

def _read_and_aggregate (result_df):
    result_df ["Combined_Field"] = "PATINET-" + result_df["Bed"] + ":" + result_df["AI Results"]
    #create a new string by joining all the values from column "Combined_Field" with a colon
    result_note_consolidated = " ".join(result_df["Combined_Field"])
    return result_note_consolidated

def generate_summary(result_df):
    with st.spinner('Generating the consolidated summary ...'):
        result_note_agg = _read_and_aggregate(result_df)
        prompt = _generate_summary_prompt(result_note_agg)
        executive_Summary= claude(prompt)   
    st.success('Consolidated summary done:',icon="✅")
    st.write(executive_Summary)
    return executive_Summary

def generate_individual_summary(source_df):
    progress_text = "Analyzing individual records, Please wait..."
    my_bar = st.progress(0, text=progress_text)
    counter = len(source_df)
    each_run = 1/counter
    percent_complete = 0
    for index, row in source_df.iterrows():
        # Get the value from the 'anonymized notes' column
        Notes = row['Anonymized_NOTE_TEXT']
        generated_prompt = _generate_prompt(Notes)
        # Call Anthropic_API.claude() with the generated prompt
        response = claude(generated_prompt)    
        # Store the response in the 'AI Results' column
        source_df.at[index, 'AI Results'] = response.lstrip()
        # subset_df = result_note[["Bed", "AI Results"]]
        my_bar.progress(percent_complete + each_run, text=progress_text)
        percent_complete += each_run
    st.success("Individual records analysis done:", icon="✅")
    st.dataframe(source_df[['Bed', 'AI Results']])
    download_results_df = source_df[['Bed', 'AI Results']]
    return download_results_df

@st.cache_data
def _convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def download_button(df):  
    csv = _convert_df(df)
    st.download_button(
    "Download",
    csv,
    "LongStayerResults.csv",
    "text/csv",
    key='download-csv'
    )
