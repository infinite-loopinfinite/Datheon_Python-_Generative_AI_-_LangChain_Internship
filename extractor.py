import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional

class WorkExperience(BaseModel):
    company: str = Field(description="Name of the company or organization")
    role: str = Field(description="Job title or role")
    duration: str = Field(description="Time period worked (e.g., '2021 - 2023' or 'Jan 2022 - Present')")
    key_achievements: List[str] = Field(description="List of key responsibilities or achievements")

class Education(BaseModel):
    degree: str = Field(description="Degree or qualification obtained")
    institution: str = Field(description="University, college, or school name")
    graduation_year: Optional[int] = Field(None, description="Year of graduation, if mentioned")

class ResumeSchema(BaseModel):
    full_name: str = Field(description="Candidate's full name")
    email: Optional[str] = Field(None, description="Candidate's email address")
    phone: Optional[str] = Field(None, description="Candidate's contact phone number")
    skills: List[str] = Field(description="List of technical and soft skills")
    experience: List[WorkExperience] = Field(description="List of work experiences")
    education: List[Education] = Field(description="Educational background")

FEW_SHOT_EXAMPLES = """
Example Input Text:
"Sarah Connor. Reach me at s.connor@sky.net or 555-0199. I graduated with a B.S. in Computer Science from MIT in 2018. Worked at Cyberdyne Systems from 2018 to 2022 as a Lead Systems Engineer where I built neural net processors and prevented system failures. Proficient in Python, C++, and Robotics."

Example Output:
{
  "full_name": "Sarah Connor",
  "email": "s.connor@sky.net",
  "phone": "555-0199",
  "skills": ["Python", "C++", "Robotics"],
  "experience": [
    {
      "company": "Cyberdyne Systems",
      "role": "Lead Systems Engineer",
      "duration": "2018 - 2022",
      "key_achievements": ["Built neural net processors", "Prevented system failures"]
    }
  ],
  "education": [
    {
      "degree": "B.S. in Computer Science",
      "institution": "MIT",
      "graduation_year": 2018
    }
  ]
}
"""

def build_prompt(document_type: str, raw_text: str, examples: str) -> str:
    prompt_template = f"""
You are an expert data extraction system. Your task is to extract structured information from an unstructured {document_type}.
Strictly adhere to the provided JSON schema. Do not invent information; if a field is missing from the text, leave it as null or an empty list.

--- FEW-SHOT EXAMPLES ---
{examples}

--- ACTUAL INPUT TO EXTRACT ---
Document Type: {document_type}
Raw Text:
\"\"\"
{raw_text}
\"\"\"
"""
    return prompt_template.strip()

def extract_structured_data(raw_text: str, document_type: str = "Resume") -> ResumeSchema:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set your API key.")
    
    # Initialize the client using the new google-genai SDK
    client = genai.Client(api_key=api_key)

    prompt = build_prompt(
        document_type=document_type,
        raw_text=raw_text,
        examples=FEW_SHOT_EXAMPLES
    )

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ResumeSchema,
            temperature=0.0,
        ),
    )

    validated_data = ResumeSchema.model_validate_json(response.text)
    return validated_data

if __name__ == "__main__":
    sample_unstructured_resume = """
    Hey there! Here's my bio/resume info. Name's Alex Rivera (alex.rivera.dev@email.com, tel: +1-415-890-2341).
    I've been coding for about 4 years now. Most recently, I was a Senior Backend Developer at FinTech Corp 
    from March 2022 until Present. While there, I migrated monolithic services to microservices using FastAPI 
    and reduced API latency by 40%. Before that (June 2020 - Feb 2022), I worked at StartupX as a Junior Python Dev, 
    building automated web scrapers and managing PostgreSQL databases.
    
    As for schooling, I did my Bachelor of Technology in Information Technology at Delhi Technological University, 
    graduating back in 2020. 
    My main tech stack includes Python, FastAPI, Docker, PostgreSQL, AWS, and Git.
    """

    try:
        extracted_resume = extract_structured_data(sample_unstructured_resume, document_type="Resume")
        print(json.dumps(extracted_resume.model_dump(), indent=2))
    except Exception as e:
        print(f"Error: {e}")
