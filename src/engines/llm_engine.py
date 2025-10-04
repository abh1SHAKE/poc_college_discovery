import os
import json
import re
from typing import List, Dict
from datetime import datetime
from ..models.college import College, Course, VerificationStatus, EvidenceStatus
import groq


class CollegeDiscoveryEngine:
    def __init__(self, api_key: str, model: str = None):
        """Initialize Groq client and model"""

        self.model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        self.client = groq.Client(api_key=api_key)

    def create_discovery_prompt(self, location: str, career_path: str) -> str:
        """Create a detailed prompt for college discovery"""

        return f"""
        You are an expert educational consultant specializing in Indian higher education.

        Task: Find colleges in {location} offering courses related to {career_path}.

        Requirements:
        1. Only include colleges physically located in {location}
        2. Focus on courses directly related to {career_path}
        3. Prefer government and well-established private institutions
        4. Include official website URLs (college domain only - .edu.in, .ac.in, .org.in)
        5. Be conservative - only include colleges you're confident exist
        6. Provide accurate, verifiable information

        Output Format (JSON):
        {{
        "colleges": [
            {{
            "name": "Exact college name",
            "city": "City name",
            "state": "State name", 
            "type": "Government|Private|Deemed University|Central University|State University",
            "website": "https://official-college-domain.ac.in",
            "confidence": 0.85,
            "courses": [
                {{
                "course_name": "Full course name (e.g., Bachelor of Technology in Computer Science)",
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
        - Maximum 15 colleges to ensure quality over quantity
        - Use confidence scores between 0.6-0.95 (be realistic)
        - If uncertain about fees/seats, omit rather than guess
        - Focus on well-known, established institutions
        - Ensure course names are specific and accurate
        - Include only verified entrance exams
        """

    async def discover_colleges(self, location: str, career_path: str) -> List[College]:
        """Discover colleges using Groq LLM"""

        prompt = self.create_discovery_prompt(location, career_path)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise educational data expert. Always return valid JSON with accurate information about Indian colleges and universities."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.1,
                top_p=0.9
            )

            content = response.choices[0].message.content.strip()

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
        """Parse LLM response into College objects"""
        
        colleges = []

        for college_data in data.get("colleges", []):
            try:
                courses = []
                for course_data in college_data.get("courses", []):
                    course = Course(
                        name=course_data.get("course_name", ""),
                        degree_level=course_data.get("degree_level", "UG"),
                        official_source_url=college_data.get("website", ""),
                        row_confidence=college_data.get("confidence", 0.5),
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
