from typing import List, Dict
import os
import fitz
import re
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

def smart_chunk_pdf(pdf_path, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    """
    Parses a PDF file page-by-page, cleans whitespace, and splits it into 
    overlapping chunks without cutting words in half.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    doc = fitz.open(pdf_path)
    all_chunks = []
    chunk_id = 1
    file_name = os.path.basename(pdf_path)

    doc_metadata = {
        "source": pdf_path,
        "title": doc.metadata.get('title', 'Unknown title'),
        "author": doc.metadata.get('author', 'Unknown author'),
        "creation_date": doc.metadata.get('creationDate', 'Unknown Date')
    }

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        # Extract page-level metadata
        page_metadata = {
            "page_number": page_num + 1,
            "total_pages": len(doc)
        }
        # Extract Heading
        headings = extract_headings(text)

        # Detect if its resume section
        section_type = detect_section_type(text, headings)

        # Extract skills
        detected_skills = extract_skills(text)

        if not text.strip():
            continue
        # Clean newlines so sentences remain unbroken
        cleaned_text = " ".join(text.split())

        start = 0
        while start < len(cleaned_text):
            end = start + chunk_size
            
            # Prevent cutting a word in half by shifting end to the next nearest space
            if end < len(cleaned_text):
                next_space = cleaned_text.find(" ", end)
                if next_space != -1 and (next_space - end) < 20:
                    end = next_space

            chunk_text = cleaned_text[start:end].strip()

            if chunk_text:
                all_chunks.append({
                    "id": f"{file_name}-chunk-{chunk_id}",
                    "text": chunk_text,
                    "page": page_num + 1,
                    "metadata": {
                        "source": doc_metadata["source"],
                        "title": doc_metadata["title"],
                        "author": doc_metadata["author"],
                        "section": headings[0] if headings else "General",
                        "section_type": section_type,
                        "detected_skills": detected_skills[:8], # Top 8 Skills
                        "chunk_size": len(chunk_text),
                        "start_position": start,
                        "end_position": end,
                        "has_skills": len(detected_skills) > 0,
                        "word_count": len(chunk_text.split()),
                        "creation_date": doc_metadata["creation_date"],
                        "page_number": page_metadata["page_number"],
                    }   
                })
                chunk_id += 1
            
            # Shift window forward by chunk size minus overlap
            start += (chunk_size - overlap)
            
    doc.close()
    return all_chunks

def extract_headings(text: str) -> List[str]:
    """
    Extract potential headings/section titles from text
    """
    heading_patterns = [
        r'^([A-Z][A-Z\s]+)$',  # ALL CAPS
        r'^(Education|Experience|Skills|Projects|Work|Employment|Summary|Objective|Certifications|Achievements)',
        r'^[A-Z][a-z]+ [A-Z][a-z]+',  # Title Case
    ]

    headings = []
    lines = text.split("\n")
    for line in lines[:12]:
        line = line.strip()
        for pattern in heading_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                headings.append(line)
                break
    return headings

def detect_section_type(text: str, heading: List[str]) -> List[str]:
    """
    Detect what type of section is
    """
    text_lower = text.lower()

    experience = ['experience', 'work', 'employment', 'job']
    education = ['education', 'university', 'college', 'degree']
    skills = ['skill', 'proficient', 'expert', 'knowledge']
    project = ['project', 'built', 'developed', 'created']
    summary =  ['summary', 'objective', 'profile']

    if any(word in text_lower for word in experience):
        return "experience"
    elif any(word in text_lower for word in education):
        return "education"
    elif any(word in text_lower for word in skills):
        return "skills"
    elif any(word in text_lower for word in project):
        return "project"
    elif any(word in text_lower for word in summary):
        return "summary"
    else:
        return "general"

def normalize_skill(skill: str) -> str:
    """Normalize common skill variations to standard names"""
    normalization_map = {
        # Programming languages
        'js': 'javascript',
        'ts': 'typescript',
        'py': 'python',
        'rb': 'ruby',
        'c plus plus': 'c++',
        'c sharp': 'c#',
        'c-sharp': 'c#',
        'csharp': 'c#',
        'golang': 'go',
        'go lang': 'go',
        
        # Frameworks & Libraries
        'reactjs': 'react',
        'react.js': 'react',
        'vuejs': 'vue',
        'vue.js': 'vue',
        'nodejs': 'node',
        'node.js': 'node',
        'expressjs': 'express',
        'express.js': 'express',
        'django rest': 'django',
        'django rest framework': 'django',
        
        # Cloud & DevOps
        'aws services': 'aws',
        'amazon web services': 'aws',
        'azure cloud': 'azure',
        'microsoft azure': 'azure',
        'google cloud': 'gcp',
        'google cloud platform': 'gcp',
        'gcloud': 'gcp',
        'docker containers': 'docker',
        'containerization': 'docker',
        'k8s': 'kubernetes',
        'kube': 'kubernetes',
        'terraform iac': 'terraform',
        'ci cd': 'ci/cd',
        'cicd': 'ci/cd',
        'ci-cd': 'ci/cd',
        'continuous integration': 'ci/cd',
        'jenkins ci': 'jenkins',
        
        # Databases
        'postgresql': 'postgres',
        'postgres db': 'postgres',
        'mongo': 'mongodb',
        'mongo db': 'mongodb',
        'redis cache': 'redis',
        'mysql database': 'mysql',
        'nosql': 'mongodb',
        
        # AI/ML
        'ml': 'machine learning',
        'ai/ml': 'ai',
        'artificial intelligence': 'ai',
        'deep neural networks': 'deep learning',
        'natural language processing': 'nlp',
        'nlu': 'nlp',
        'llm': 'ai',
        'large language models': 'ai',
        'genai': 'ai',
        'generative ai': 'ai',
        
        # Web Technologies
        'html5': 'html',
        'css3': 'css',
        'scss': 'sass',
        'tailwindcss': 'tailwind',
        'tailwind css': 'tailwind',
        'restful api': 'rest api',
        'restful': 'rest api',
        'graphql api': 'graphql',
        'microservices architecture': 'microservices',
        'micro services': 'microservices',
        
        # Version Control
        'github actions': 'github',
        'gitlab': 'git',
        'bitbucket': 'git',
        'version control': 'git',
        
        # Methodologies
        'agile methodology': 'agile',
        'scrum master': 'scrum',
        'kanban': 'agile',
        'devops': 'ci/cd',
    }
    
    skill_lower = skill.lower().strip()
    return normalization_map.get(skill_lower, skill_lower)

def extract_skills(text: str) -> List[str]:
    """Extract and normalize skills from text"""
    static_skills = [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go',
        'react', 'angular', 'vue', 'node', 'express', 'django', 'flask',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
        'sql', 'mongodb', 'postgres', 'mysql', 'redis',
        'git', 'github', 'ci/cd', 'jenkins', 'agile', 'scrum',
        'machine learning', 'ai', 'deep learning', 'nlp',
        'html', 'css', 'sass', 'tailwind',
        'rest api', 'graphql', 'microservices'
    ]
    
    text_lower = text.lower()
    extracted_skills = set()
    
    # 1. Static skill matching
    for skill in static_skills:
        if skill in text_lower:
            extracted_skills.add(skill)
    
    # 2. Skills section extraction
    skills_pattern = r'(?:SKILLS|TECHNICAL SKILLS|TECHNOLOGIES|COMPETENCIES)[\s:]*\n(.*?)(?:\n\n|\n[A-Z][A-Z\s]+:|\Z)'
    match = re.search(skills_pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        skills_text = match.group(1)
        raw_skills = re.split(r'[,•·|/\n]', skills_text)
        for skill in raw_skills:
            skill = skill.strip().lower()
            if skill and len(skill) >= 2:
                extracted_skills.add(normalize_skill(skill))  # Normalize here
    
    # 3. Contextual pattern extraction
    contextual_patterns = [
        r'(?:proficient|experienced|skilled)\s+(?:in|with)\s+([\w\s+#.-]+?)(?:[,.]|\band\b)',
        r'(?:worked\s+(?:with|on)|used|developed)\s+([\w\s+#.-]+?)(?:[,.]|\band\b)',
        r'(?:knowledge\s+of|familiar\s+with)\s+([\w\s+#.-]+?)(?:[,.]|\band\b)',
    ]
    
    for pattern in contextual_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            skill = match.group(1).strip().lower()
            if len(skill.split()) <= 4 and len(skill) >= 2:
                extracted_skills.add(normalize_skill(skill))  # Normalize here
    
    # 4. Final cleanup: remove duplicates after normalization
    return list(extracted_skills)


