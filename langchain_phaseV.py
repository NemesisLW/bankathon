# %% [markdown]
# # Langchain Implementations

# %%
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain

# %% [markdown]
# # Job Evaluation - Suggested Job Description

# %%
title = input("Job title:") # Input Box for Title
description = input("description:") # For Description

# %%
llm = OpenAI(temperature=0.6,openai_api_key="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", model_name="gpt-3.5-turbo-16k")

# %%
# Button to evaluate and get enchanced Job Description, it will run the following code.

evaluationtemplate ="""You are an AI assistant tasked with assisting a hiring manager in enhancing job descriptions provided by the HR. The HR will provide you with a job title and description, and your goal is to score the job description based on the job title and provide recommendations for improvements. You will then give the HR the option to either continue with the original version or incorporate the suggested changes.

To accomplish this, you will see the following information:

Input:
- Job Title: {title}
- Job Description: {description}

Your output should be in the form of recommendations and proposed changes to the job description. You can suggest improvements in language, emphasize important skills or qualifications, or provide additional details that would enhance the appeal of the job description.

Remember to be respectful and tactful in your recommendations, while also demonstrating your superior technical knowledge to provide valuable enhancements.
"""
eval_template = PromptTemplate(input_variables=["title", "description"], template=evaluationtemplate)
evaluationChain = LLMChain(llm = llm, prompt=eval_template, output_key="job_description_evaluation")

updatedDesctemplate ="""Job Description: {description}

Proposed Enhancements and Recommendation:
{job_description_evaluation}

Updated Description:

"""
updateddesc_template = PromptTemplate(input_variables=["description", "job_description_evaluation"], template=updatedDesctemplate)
updatedChain = LLMChain(llm=llm, prompt=updateddesc_template, output_key="updated_job_description")

enchancement_chain = SequentialChain(chains=[evaluationChain,updatedChain], input_variables=["title", "description"], output_variables=["job_description_evaluation","updated_job_description"], verbose=True)

results = enchancement_chain({"title":title, "description": description })

# %%
# First display the updated job description
#then,
# Two Buttons, for two options
#    1. Keep the changes 2. Discard Changes.
print(results["updated_job_description"])

# %% [markdown]
# # Shortlist Chain

# %%
# Start Evaluation of candidates and shortlist them Button

import PyPDF2
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

pdf_paths = filedialog.askopenfilenames(title="Select PDF files", filetypes=[("PDF files", "*.pdf")])

extracted_text = ""

for pdf_path in pdf_paths:
    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        extracted_text += page.extract_text() + '\n'

    pdf_file.close()

print(extracted_text)

# %%
import os

from langchain.llms import OpenAI
llm=OpenAI(temperature=0.6,openai_api_key="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", model_name="gpt-3.5-turbo-16k")
from langchain.prompts import PromptTemplate

prompt_template_name=PromptTemplate(
    
    input_variables=['CV_text'],
    template="""The Following is a series of a cv texts in a continuous manner of different candidates:
{CV_text}

Your task is to extract the key details, such as [name, email, skills, experience and projects] for each candidate. Remember the text has details of multiple candidate.

You should follow the following the rules while extracting details:
1. Do not include schooling details in experiences.
2. Try to summarise the details whenever you can.

Your Response should follow the following format:

[{{"name": "Name of first candidate", "email": "email of first candidate", "skills": "skills of first candidate", "experiences of first candidate", "projects": "Summary of project details of the candidate","extras":"If there is any other relevant information"}},]

Make sure you do not add any false information. If you do not find the required values for the keys, Say "Not applicable." 

Remember, you must not output any other text other than the JSON object.
"""
    
)
prompt_template_name.format(CV_text=extracted_text)



# %%
from langchain.chains import LLMChain

chain=LLMChain(llm=llm,prompt=prompt_template_name)
response=chain.run(extracted_text)

# %%
print(response)

# %%
import json

candidate_dict = json.loads(response)
candidate_dict

# %%
shortlisttemplate= """As an Expert Resume Screener, you have excellent skills to screen and shortlist candidates based on the skills, experience,  previous projects and education. You are provided with a job description and details of the candidates as a JSON object. 

Now your task is to:

1. Rank the CVs according to their alignment with the job requirements and shortlist candidates. 
2. Provide additional information in short bullet points on each shortlisted candidate separately.

The following is an example of how you should format your response as JSON object:
Example:
{{"CV_Ranking": ["Name of Candidate No.3", "Name of Candidate No.5", "Name of Candidate No.2"],
  "Additional_Information": [
        {{"name": "Name of Candidate No.3","email": "email of Candidate No.3", "info": ["Additional Information 1 about skills", "Additional Information 2 about experiences", "Additional Information 3 about Projects"],}},
        {{"name": "Name of Candidate No.5","email": "email of Candidate No.5", "info": ["Additional Information 1 about skills", "Additional Information 2 about experiences", "Additional Information 3 about Projects", "any other additional information"],}},
        {{"name": "Name of Candidate No.2","email": "email of Candidate No.2", "info": ["Additional Information 1 about skills", "Additional Information 2 about experiences", "Additional Information 3 about Projects"],}}]}}
        
This is the end of Example.

Now, here is the Job Description: 
{description}

The following the details of the candidates who applied: 
{candidates}

procced to execute your task. Do not generate any extra text other than the JSONs."""

shortlist_template = PromptTemplate(input_variables=["description", "candidates"], template=shortlisttemplate)
shortlistChain = LLMChain(llm=llm, prompt=shortlist_template, output_key="shortlist")

shortlist_chain = SequentialChain(chains=[shortlistChain], input_variables=["description", "candidates"], output_variables=["shortlist"], verbose=True)
shortlist = shortlist_chain({"description": description, "candidates": candidate_dict })

# %%
print(shortlist['shortlist'])

# %%
short=shortlist['shortlist']

# %%
short

# %%
import ast
data_dict = ast.literal_eval(short)
print(data_dict)

# %%
short=data_dict

# %%
short

# %% [markdown]
# # Emails Chain

# %%
# Display the details of the shortlisted candidates
# Then, Button to send emails to them.

contact_info = []

for candidate in short['Additional_Information']:
    contact_info.append({
        'name': candidate['name'].split()[0],  # Extracting the first name
        'email': candidate['email']
    })

print(contact_info)


# %%
emails = []

for person_info in short["Additional_Information"]:
    emails.append(person_info["email"])
emails

# %%
names =[]
for person_info in short["Additional_Information"]:
    names.append(person_info["name"])
names

# %%
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.agents import AgentType
from langchain.utilities.zapier import ZapierNLAWrapper

# %%
zapier = ZapierNLAWrapper(zapier_nla_api_key="sk-ak-vYq75pJK5wHOwgchYPQWrJAQua")
actions = zapier.list()

# %%
actions

# %%
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
agent = initialize_agent(toolkit.get_tools(), llm, agent="zero-shot-react-description", verbose=True)

# %%
agent.run(f"""Your task is to send personalized congratulatory emails to the selected candidates who are {names}, informing them about their selection and the next steps in the hiring process.

Please craft individual emails for each candidate, addressing them by their name. You can find the names of the caand including specific details about their selection and the next steps. Your emails should be professional, concise, and well-written, demonstrating enthusiasm for their selection and providing clear instructions on what they need to do next.

Please note that each email should be unique and tailored to the individual candidate. You should avoid using any generic or template language. Instead, personalize each email by mentioning specific qualifications, experiences, or accomplishments that stood out during the selection process. Additionally, feel free to include any relevant information about the company, team, or role that may be of interest to the candidate.

You may consult the following JSON object to gain specific information about the email addresses of each candidate: 

{contact_info}

Ensure that the emails are error-free, have a professional tone, and are formatted correctly. Check the names and emails of the candidates to ensure accuracy before sending the emails.

Your goal is to make each candidate feel appreciated, valued, and excited about the next steps in the hiring process.

NO NEED to CC anyone. Make sure you do not include "[Candidate's Email]" in the "To" section.
Make sure you have sent the emails to every eligible candidates selected.
""")

# %% [markdown]
# # Screening Questionnaire

# %%
# Generate Screening Questionnaire
screeningtemplate = """You are a member of the hiring committee of your company. Your task is to develop screening questions for each candidate, considering different levels of importance or significance assigned to the job description and the candidate's CV. You will have all the necessary information about the job title, job description and all relevant information about the candidate. 

Here are the Details:
Job title: {title}
Job description: {description}
Candidate Details: {shortString}

Your Response should follow the following format:
[{{"id":1, "Question":"Your Question will go here"}},
{{"id":2, "Question":"Your Question will go here"}},
{{"id":3, "Question":"Your Question will go here"}}]

There should be at least 10 questions. Do not output anything other than the JSON object."""

screen_template = PromptTemplate(input_variables=["title", "description", "shortString"], template=screeningtemplate)
screenChain = LLMChain(llm = llm, prompt=screen_template, output_key="questions")

screen_chain = SequentialChain(chains=[screenChain], input_variables=["title", "description", "shortString"], output_variables=["questions"], verbose=True)

ques = screen_chain({"title":title, "description": description, "shortString": short })

# %%
import json

# %%
questionnaire = json.loads(ques["questions"])
questionnaire

# %%
# This section needs modification, we need to record the response of each candidate separately. That is not implemented yet.
answers = []

for question in questionnaire:
    user_response = input(question['Question'] + '\n')
    answer_entry = {'id': question['id'], 'Answer': user_response}
    answers.append(answer_entry)
answers

# %%
sheet = ""
for answer_entry in answers:
    question_id = answer_entry['id']
    question = next(q['Question'] for q in questionnaire if q['id'] == question_id)
    answer = answer_entry['Answer']
    print(f"Question {question_id}:\n{question}\nAnswer {question_id}:\n{answer}\n")
    sheet += f"Question {question_id}:\n{question}\nAnswer {question_id}:\n{answer}\n"

# %%
sheet

# %% [markdown]
# # First Round

# %%
finalevaltemplate = """"You are a member of the hiring committee of your company. Shortlisted Candidates have been tested with screening questions. you will provided with the Questions and the answers given by the candidate for each question. Your task is to evaluate their performance  for consideration in the next in-person round.
The following is the Answer Script with questions and their answers by the candidate:
{sheets}
Your task is to evaluate the answers for each of the question and mark the each answer between 0-10. Then you should get the total marks obtained by the candidate and then you should output the names of the top 2 candidates who should be considered for next round of conversation. 

Your Response should follow the following format:
[{{"Name":"Name of the candidate", "Marks":"Total Marks goes here", "shortlisted": true}},
{{"Name":Name of the candidate, "Marks":"Total Marks goes here", "shortlisted": true}},
{{"Name":Name of the candidate, "Marks":"Total Marks goes here", "shortlisted": false}},]

Do not output anything other than the JSON object."""

finaleval_template = PromptTemplate(input_variables=["sheets"], template=finalevaltemplate)
finalevalChain = LLMChain(llm = llm, prompt=finaleval_template, output_key="selected")

finaleval_chain = SequentialChain(chains=[finalevalChain], input_variables=["sheets"], output_variables=["selected"], verbose=True)

finaleval_chain({"sheets": sheet})

# %%



