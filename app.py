from dotenv import load_dotenv
import os

from langchain.llms import OpenAI
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.callbacks import StreamlitCallbackHandler


from description_evaluator import evaluator
from shortlister import rank_and_shortlist
from send_email import send_emails
from screening import screeing_test
from firstRound import final_eval

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
zapier_key = os.getenv("ZAPIER_API_KEY")

llm = OpenAI(temperature=0.6, openai_api_key=api_key, model_name="gpt-3.5-turbo-16k")
zapier = ZapierNLAWrapper(zapier_nla_api_key=zapier_key)

title = input("Job Title:")
description = input("Job Description:")

print("Evaluating the job description...")
results = evaluator(title, description, llm=llm)


print("Updated job description: \n")
print(results["updated_job_description"])

print("Before moving on to next step...")

swtich = int(input("Type: 1 to update the job description, 0 to discard changes."))
if swtich == 1:
    description = results["updated_job_description"]

print("Ranking and Shortlisting Candidates....\n")
short = rank_and_shortlist(description, llm=llm)

print(short)

contact_info = []

for candidate in short['Additional_Information']:
    contact_info.append({
        'name': candidate['name'].split()[0],  # Extracting the first name
        'email': candidate['email']
    })

print(contact_info)

emails = []

for person_info in short["Additional_Information"]:
    emails.append(person_info["email"])

names =[]
for person_info in short["Additional_Information"]:
    names.append(person_info["name"])


send_emails(llm=llm, zapier=zapier, names=names, contact_info=contact_info)

sheet = screeing_test(llm=llm, title=title, description=description, short=short)
