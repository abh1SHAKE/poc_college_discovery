import openai
import json
import re
from typing import List, Dict
from datetime import datetime
from ..models.college import College, Course, VerificationStatus, EvidenceStatus, DegreeLevel

class CollegeDiscoveryEngine:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

        def create_discovery_prompt(self, location: str, career_path: str) -> str:
            return f"""
                You are an expert educational consultant specializing in Indian higher education.

                Task: Find colleges in {location} offering courses related to {career_path}.

                Requirements:
                1. Only include colleges physically located in {location}
                2. Focus on courses directly related to {career_path}
                3. Prefer government and well-established private institutions
                4. Include official website URLs (college domain only)

                Output Format (JSON):
                {{
                "colleges": [
                    {{
                    "name": "Exact college name",
                    "city": "City name",
                    "state": "State name", 
                    "type": "Government|Private|Deemed University|Central University",
                    "website": "https://official-college-domain.ac.in",
                    "confidence": 0.85,
                    "courses": [
                        {{
                        "course_name": "Full course name",
                        "degree_level": "UG|PG|Diploma|Certificate|PhD",
                        "duration": "4 years",
                        "annual_fees": "₹1,00,000",
                        "seats": 120,
                        "entrance_exams": ["JEE Main", "State CET"],
                        "specializations": ["AI/ML", "Data Science"]
                        }}
                    ]
                    }}
                ]
                }}

                Important Guidelines:
                - Only include colleges you are confident exist
                - Use official .edu.in, .ac.in domains
                - If uncertain about any detail, omit it rather than guess
                - Maximum 20 colleges to ensure quality
                - Be conservative with confidence scores
            """
        
        async def discover_colleges(self, location: str, career_path: str) -> List[College]:
            prompt = self.create_discovery_prompt(location, career_path)

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise educational data expert. Always return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )

                content = response.choices[0].message.content

                # Extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if not json_match:
                    raise ValueError("No valid JSON found in response")
                
                data = json.loads(json_match.group())
                return self._parse_colleges(data, location, career_path)
            
            except Exception as e:
                print(f"Error in LLM discovery: {e}")
                return []
            

        def _parse_colleges(self, data: Dict, location: str, career_path: str) -> List[College]:
            colleges = []

            for college_data in data.get("colleges", []):
                try:
                    courses = []
                    for course_data in college_data.get("courses", []):
                        course = Course(
                            course_name=course_data.get("course_name", ""),
                            degree_level=course_data.get("degree_level", "UG"),
                            official_source_url=college_data.get("website", ""),
                            row_confidence=college_data.get("webite", ""),
                            duration=course_data.get("duration"),
                            annual_fees=course_data.get("annual_fees"),
                            seats=course_data.get("seats"),
                            entrance_exams=course_data.get("entrance_exams", []),
                            specializations=course_data.get("specializations", [])
                        )

                        courses.append(course)

                    college = College(
                        name=college_data.get("name", ""),
                        city=college_data.get("city", ""),
                        state=college_data.get("state", ""),
                        type=college_data.get("type", ""),
                        website=college_data.get("website", ""),
                        overall_confidence=college_data.get("confidence", 0.5),
                        last_collected=datetime.now(),
                        verification_status=VerificationStatus.DRAFT,
                        evidence_status=EvidenceStatus.PENDING_VERIFICATION,
                        courses=courses
                    )

                    colleges.append(college)

                except Exception as e:
                    print(f"Error parsing college data: {e}")
                    continue

            return colleges