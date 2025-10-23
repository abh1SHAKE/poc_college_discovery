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

    def create_college_list_prompt(self, location: str) -> str:
        """Create prompt for discovering colleges (First step)"""
        return f"""
                You are an expert educational consultant specializing in Indian higher education.

                Task: Find ALL colleges and universities in {location}.

                Requirements:
                1. Only include colleges physically located in {location}
                2. Include ALL types: Government, Private, Deemed, Central, State Universities
                3. Include ALL streams: Engineering, Medical, Arts, Commerce, Science, Management, etc.
                4. Include official website URLs (college domain only - .edu.in, .ac.in, .org.in)
                5. Be comprehensive - aim for 40-60 colleges if available in the location
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
                    "confidence": 0.85
                    }}
                ]
                }}

                Important Guidelines:
                - Focus on well-known, established institutions
                - Use confidence scores between 0.6-0.95 (be realistic)
                - Prioritize colleges with official websites
                - Include as many colleges as possible from {location}
                - Do NOT include course information (that will be fetched separately)
                """

    def create_course_discovery_prompt(self, college_name: str, college_website: str, career_path: str = None) -> str:
        """Create prompt for discovering courses for a specific college (Second step)"""
        career_filter = f"\n3. Focus on courses related to: {career_path}" if career_path else ""
        
        return f"""
                You are an expert educational consultant specializing in Indian higher education.

                Task: Find ALL courses offered by {college_name}.

                College Website: {college_website}

                Requirements:
                1. List ALL undergraduate and postgraduate courses offered
                2. Include certificates, diplomas, and doctoral programs{career_filter}
                4. Provide accurate course details
                5. Include entrance exam information
                6. Be comprehensive - include all available programs

                Output Format (JSON):
                {{
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

                Important Guidelines:
                - If uncertain about fees/seats, omit rather than guess
                - Ensure course names are specific and accurate
                - Include only verified entrance exams
                - List all major specializations available
                - Be thorough - this is the only chance to capture course data for this college
                """

    async def discover_colleges(self, location: str, career_path: str = None, 
                               progress_callback=None) -> List[College]:
        """
        Two-step discovery process:
        Step 1: Discover colleges by location
        Step 2: For each college, discover all courses
        """
        # Step 1: Discover colleges
        if progress_callback:
            progress_callback("step1_start", {"location": location})
        
        colleges_basic = await self._discover_colleges_list(location)
        
        if not colleges_basic:
            return []
        
        if progress_callback:
            progress_callback("step1_complete", {"count": len(colleges_basic)})
        
        # Step 2: Discover courses for each college
        colleges_with_courses = []
        total = len(colleges_basic)
        
        for idx, college_basic in enumerate(colleges_basic):
            if progress_callback:
                progress_callback("step2_progress", {
                    "current": idx + 1,
                    "total": total,
                    "college_name": college_basic.name
                })
            
            courses = await self._discover_college_courses(
                college_basic.name,
                college_basic.website,
                career_path
            )
            
            college_basic.courses = courses
            colleges_with_courses.append(college_basic)
        
        if progress_callback:
            progress_callback("step2_complete", {"count": len(colleges_with_courses)})
        
        # Filter by career path if specified (keep colleges with matching courses)
        if career_path:
            colleges_with_courses = [
                c for c in colleges_with_courses 
                if len(c.courses) > 0
            ]
        
        return colleges_with_courses

    async def _discover_colleges_list(self, location: str) -> List[College]:
        """Step 1: Discover list of colleges"""
        prompt = self.create_college_list_prompt(location)

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
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if not json_match:
                raise ValueError("No valid JSON found in response")

            data = json.loads(json_match.group())
            return self._parse_colleges_basic(data, location)

        except Exception as e:
            print(f"Error in college list discovery: {e}")
            return []

    async def _discover_college_courses(self, college_name: str, 
                                       college_website: str,
                                       career_path: str = None) -> List[Course]:
        """Step 2: Discover courses for a specific college"""
        prompt = self.create_course_discovery_prompt(college_name, college_website, career_path)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise educational data expert. Always return valid JSON with accurate course information."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000,
                temperature=0.1,
                top_p=0.9
            )

            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if not json_match:
                print(f"No valid JSON found for {college_name}")
                return []

            data = json.loads(json_match.group())
            return self._parse_courses(data, college_website)

        except Exception as e:
            print(f"Error discovering courses for {college_name}: {e}")
            return []

    def _parse_colleges_basic(self, data: Dict, location: str) -> List[College]:
        """Parse basic college information (Step 1)"""
        colleges = []

        for college_data in data.get("colleges", []):
            try:
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
                    courses=[]  # Populated in Step 2
                )
                colleges.append(college)

            except Exception as e:
                print(f"Error parsing college data: {e}")
                continue

        return colleges

    def _parse_courses(self, data: Dict, college_website: str) -> List[Course]:
        """Parse course information (Step 2)"""
        courses = []

        for course_data in data.get("courses", []):
            try:
                course = Course(
                    name=course_data.get("course_name", ""),
                    degree_level=course_data.get("degree_level", "UG"),
                    official_source_url=college_website,
                    row_confidence=0.7,
                    duration=course_data.get("duration"),
                    annual_fees=course_data.get("annual_fees"),
                    seats=course_data.get("seats"),
                    entrance_exams=course_data.get("entrance_exams", []),
                    specializations=course_data.get("specializations", [])
                )
                courses.append(course)

            except Exception as e:
                print(f"Error parsing course data: {e}")
                continue

        return courses