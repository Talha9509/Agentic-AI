import os
from dotenv import load_dotenv
from pathlib import Path
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader
import time

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key is req")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-120b"

JDInput = input("Enter the Job Description: \n")
ResumePath = input("Enter the Path to Resume: ")

def getStructuredJD(JDInput):
    class JobDescription(BaseModel):
        role: str | None = None
        reqSkills: list[str] | None = None
        preferredSkills: list[str] | None = None
        minExp: float | None = None
        educationalReq: list[str] | None = None
        responsibilities: list[str] | None = None
        
    JobDescriptionSchema = JobDescription.model_json_schema()

    response_format = {
        "type": "json_object"
    }

    system_prompt = f"""
    You are an expert HR assistant.
    Your job is to analyze job descriptions and extract structured information from them.

    Return ONLY valid JSON matching this schema:
    {JobDescriptionSchema}
    
    IMPORTANT:
    Do NOT return the schema itself.
    Do NOT return fields like "properties", "title" or "type".
    Fill the schema with actual information extracted from the job description.

    If minimum experience is not mentioned, return null.
    If information for a list is missing, return an empty list.
    Do not invent information.
    """

    message_system = {
        "role": "system",
        "content": system_prompt
    }

    user_prompt = f"""
    Analyze the following job description:
    {JDInput}
    """

    message_user = {
        "role": "user",
        "content": user_prompt
    }

    messages = [message_system, message_user]

    response = client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    structuredJD = response.choices[0].message.content
    print()
    print("-----------------")
    print()
    print(structuredJD)
    return structuredJD

def ParseResume(ResumePath):
    pdfOrdoc = ResumePath.split(".")[-1]
    if(pdfOrdoc == "pdf"):
        reader = PdfReader(ResumePath)
        resumeLength = len(reader.pages)
        allPagesContent = ""
        for i in range(resumeLength):
            page = reader.pages[i]
            if "/Annots" in page:
                for annotation in page["/Annots"]:
                    subtype = annotation.get_object()["/Subtype"]
                    if(subtype == "/Link"):
                        # print(annotation.get_object())
                        action = annotation.get_object()["/A"]
                        if action["/Type"] == "/Action" and action["/URI"]:
                            link = action["/URI"]
                            # print(link)
                            allPagesContent = allPagesContent + "\n" + link
            pageContent = page.extract_text()
            allPagesContent = allPagesContent + "\n" + pageContent
    elif(pdfOrdoc == "docs" or "docx"):
        return "No support for docs"
    print()
    print("------------------------")
    print()
    print(allPagesContent)
    return allPagesContent  

def getStructuredResume():
    parsedResume = ParseResume(ResumePath) 
    
    class Exp(BaseModel):
        companyName: str
        role: str
        duration: str
        description: list[str]
        skillsUsed: list[str]
        
    class Project(BaseModel):
        title: str
        toolsTechUsed: list[str]
        liveLink: str | None = None
        githubLink: str | None = None
        description: list[str]
        
    class Certification(BaseModel):
        title: str
        description: str
        link: str
        
    class Achievement(BaseModel):
        title: str
        description: str
        link: str | None = None
        
    class Resume(BaseModel):
        name: str
        email: str | None = None
        phone: str | None = None
        totalExp: int
        skills: list[str]
        experiences: list[Exp] | None = None
        certifications: list[Certification] | None = None
        projects: list[Project] | None = None
        achievements: list[Achievement] | None = None
    
    ResumeSchema = Resume.model_json_schema()

    response_format = {
        "type": "json_object"
    }

    system_prompt = f"""
    You are an expert resume parser.
    Extract information from the resume based on its meaning, not only based on exact section headings.

    Different resumes may use different headings.
    For example:
    - Experience
    - Professional Experience
    - Work History
    - Employment
    - Internships

    These may all contain relevant experience.

    Skills may also appear in the skills section, work experience,
    internships or projects.

    Return ONLY valid JSON matching this schema:
    {ResumeSchema}

    Important rules:
    1. Do not invent information.
    2. If a value is not available, return null.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """

    message_system = {
        "role": "system",
        "content": system_prompt
    }

    text = parsedResume
    user_prompt = f"""
    Parse the following resume:
    {text}
    """

    message_user = {
        "role": "user",
        "content": user_prompt
    }

    messages = [message_system, message_user]

    response = client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    structuredResume = response.choices[0].message.content
    print()
    print("-----------------")
    print()
    print(structuredResume)
    return structuredResume

def finalScore():
    structuredJD= getStructuredJD(JDInput)
    time.sleep(2)
    structuredResume = getStructuredResume()
    
    class finalScore(BaseModel):
        score: int
        name: str
        matchingSkills: list[str] | None = None
        missingSkills: list[str] | None = None
        reason: str
        
    FinalScoreSchema = finalScore.model_json_schema()
    
    response_format = {
        "type": "json_object"
    }
    
    system_prompt = f"""
    You are senior HR who needs to give scores for the resume based on the Job description based on the schema and give a json output.
    {FinalScoreSchema}
    """
    
    message_system = {
        "role": "system",
        "content": system_prompt
    }
    
    user_prompt = f"""
    This is the Resume: {structuredResume} and this is the Job Description: {structuredJD}.
    """
    
    message_user = {
        "role": "user",
        "content": user_prompt
    }
    
    messages = [message_system, message_user]
    response = client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    finalScores = response.choices[0].message.content
    return finalScores
    

finalScores = finalScore()
print()
print("--------------------------")
print()
print(finalScores)
