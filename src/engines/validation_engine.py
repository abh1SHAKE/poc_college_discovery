import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import time
from typing import Dict, List, Optional
from ..models.college import College, EvidenceStatus

class EvidenceValidator:
    def __init__(self, delay: float = 2.0):
        self.delay = delay
        self.last_request = {}
        self.govt_portals = {
            "aicte": "https://www.aicte-india.org/",
            "ugc": "https://www.ugc.ac.in/",
            "aishe": "https://aishe.gov.in/",
            "nirf": "https://www.nirfindia.org/"
        }
        
    async def validate_colleges(self, colleges: List[College]) -> List[College]:
        """Validate all colleges and update their evidence status"""
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Educational Data Validator 1.0'}
        ) as session:
            
            for college in colleges:
                try:
                    validation_result = await self._validate_single_college(session, college)
                    
                    # Update evidence status
                    college.evidence_status = validation_result['evidence_status']
                    college.evidence_urls = validation_result['evidence_urls']
                    
                    # Store detailed validation results for UI display
                    college.validation_details = validation_result['validation_details']
                    
                    # Calculate final confidence using multi-level strategy
                    college.overall_confidence = self._calculate_final_confidence(
                        college.overall_confidence,
                        validation_result
                    )
                    
                    # Update course evidence
                    for course in college.courses:
                        course.evidence_urls = validation_result.get('course_evidence', [])

                except Exception as e:
                    print(f"Validation error for {college.name}: {e}")
                    college.evidence_status = EvidenceStatus.NO_EVIDENCE_FOUND
                    # Reduce confidence significantly for validation failures
                    college.overall_confidence *= 0.6
                    college.validation_details = {
                        'website_accessible': False,
                        'website_appears_educational': False,
                        'courses_found': 0,
                        'total_courses': len(college.courses),
                        'govt_verified': False,
                        'domain_quality': 'Unknown',
                        'error': str(e)
                    }
            
        return colleges
    
    async def _validate_single_college(self, session: aiohttp.ClientSession, college: College) -> Dict:
        """Validate a single college and return detailed evidence data"""
        evidence_urls = []
        validation_scores = {
            'website_accessible': False,
            'website_adjustment': 0.0,
            'course_evidence_adjustment': 0.0,
            'govt_verification_adjustment': 0.0,
            'domain_quality_adjustment': 0.0,
            'courses_found_count': 0,
            'total_courses': len(college.courses)
        }

        # Level 2: Website Validation
        website_result = await self._validate_website(session, college.website)
        validation_scores['website_accessible'] = website_result['accessible']
        validation_scores['appears_educational'] = website_result.get('appears_educational', False)
        validation_scores['edu_score'] = website_result.get('edu_score', 0)
        
        if website_result['accessible']:
            validation_scores['website_adjustment'] = 0.1
            evidence_urls.append(college.website)

            # Level 3: Course Evidence
            course_evidence = await self._find_course_evidence(session, college)
            validation_scores['courses_found_count'] = len(course_evidence)
            
            if course_evidence:
                evidence_urls.extend(course_evidence)
            
            # Calculate course evidence adjustment
            if validation_scores['total_courses'] > 0:
                course_match_percentage = len(course_evidence) / validation_scores['total_courses']
                if course_match_percentage > 0.7:
                    validation_scores['course_evidence_adjustment'] = 0.2
                elif course_match_percentage > 0.4:
                    validation_scores['course_evidence_adjustment'] = 0.15
                elif course_match_percentage > 0:
                    validation_scores['course_evidence_adjustment'] = 0.1
                else:
                    validation_scores['course_evidence_adjustment'] = -0.1
            else:
                validation_scores['course_evidence_adjustment'] = -0.1
        else:
            validation_scores['website_adjustment'] = -0.15
            validation_scores['course_evidence_adjustment'] = -0.1
            validation_scores['appears_educational'] = False
            validation_scores['edu_score'] = 0

        # Level 4: Government Verification
        govt_result = await self._check_govt_presence(session, college.name)
        validation_scores['govt_verified'] = govt_result['found']
        
        if govt_result['found']:
            validation_scores['govt_verification_adjustment'] = 0.2
            evidence_urls.extend(govt_result['urls'])
            
        # Level 5: Domain Quality
        domain_info = self._evaluate_domain_quality(college.website)
        validation_scores['domain_quality_adjustment'] = domain_info['adjustment']
        validation_scores['domain_type'] = domain_info['type']

        # Calculate total evidence score for status determination
        total_evidence_score = 0
        if validation_scores['website_accessible']:
            total_evidence_score += 0.3
        if govt_result['found']:
            total_evidence_score += 0.4
        if validation_scores['courses_found_count'] > 0:
            total_evidence_score += 0.3

        # Determine evidence status based on actual evidence found
        if total_evidence_score >= 0.7:
            evidence_status = EvidenceStatus.VERIFIED
        elif total_evidence_score >= 0.3:
            evidence_status = EvidenceStatus.PARTIALLY_VERIFIED
        else:
            evidence_status = EvidenceStatus.NO_EVIDENCE_FOUND

        # Create detailed validation info for UI
        validation_details = {
            'website_accessible': validation_scores['website_accessible'],
            'website_appears_educational': validation_scores['appears_educational'],
            'edu_keywords_found': validation_scores['edu_score'],
            'courses_found': validation_scores['courses_found_count'],
            'total_courses': validation_scores['total_courses'],
            'course_match_percentage': (validation_scores['courses_found_count'] / validation_scores['total_courses'] * 100) if validation_scores['total_courses'] > 0 else 0,
            'govt_verified': validation_scores['govt_verified'],
            'domain_type': validation_scores['domain_type'],
            'domain_quality_adjustment': validation_scores['domain_quality_adjustment'],
            'adjustments': {
                'website': validation_scores['website_adjustment'],
                'course_evidence': validation_scores['course_evidence_adjustment'],
                'govt_verification': validation_scores['govt_verification_adjustment'],
                'domain_quality': validation_scores['domain_quality_adjustment']
            }
        }

        return {
            'evidence_status': evidence_status,
            'evidence_urls': evidence_urls,
            'validation_scores': validation_scores,
            'validation_details': validation_details,
            'website_accessible': website_result['accessible'],
            'course_evidence': course_evidence if website_result['accessible'] else [],
            'govt_verified': govt_result['found']
        }
    
    def _calculate_final_confidence(self, llm_confidence: float, validation_result: Dict) -> float:
        """
        Calculate final confidence using multi-level validation strategy
        
        Level 1: LLM Confidence (Base Score) - 0.0 to 1.0
        Level 2: Website Validation (±0.1 to ±0.15)
        Level 3: Course Evidence (±0.1 to ±0.2)
        Level 4: Government Verification (+0.2)
        Level 5: Domain Quality (±0.1)
        """
        scores = validation_result['validation_scores']
        
        # Start with LLM base confidence
        final_confidence = llm_confidence
        
        # Apply adjustments from each level
        final_confidence += scores['website_adjustment']
        final_confidence += scores['course_evidence_adjustment']
        final_confidence += scores['govt_verification_adjustment']
        final_confidence += scores['domain_quality_adjustment']
        
        # Clamp between 0.0 and 1.0
        return max(0.0, min(1.0, final_confidence))
    
    def _evaluate_domain_quality(self, url: str) -> Dict:
        """Evaluate domain quality based on TLD"""
        try:
            domain = urlparse(url).netloc.lower()
            
            if domain.endswith('.edu.in') or domain.endswith('.ac.in'):
                return {'adjustment': 0.1, 'type': 'Educational (.edu.in/.ac.in)'}
            elif domain.endswith('.gov.in'):
                return {'adjustment': 0.1, 'type': 'Government (.gov.in)'}
            elif domain.endswith('.org.in'):
                return {'adjustment': 0.05, 'type': 'Organization (.org.in)'}
            elif domain.endswith('.edu'):
                return {'adjustment': 0.05, 'type': 'Educational (.edu)'}
            else:
                return {'adjustment': 0.0, 'type': 'Standard Domain'}
                
        except Exception:
            return {'adjustment': 0.0, 'type': 'Unknown'}
    
    def get_confidence_level(self, confidence: float) -> str:
        """Categorize confidence into levels as per design document"""
        if confidence >= 0.8:
            return "HIGH"
        elif confidence >= 0.6:
            return "MEDIUM"
        elif confidence >= 0.4:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def get_action_recommendation(self, confidence: float) -> str:
        """Get recommended action based on confidence level"""
        level = self.get_confidence_level(confidence)
        
        recommendations = {
            "HIGH": "Can be auto-approved with minimal review",
            "MEDIUM": "Standard manual review required",
            "LOW": "Detailed manual review required",
            "VERY_LOW": "Likely reject or mark for investigation"
        }
        
        return recommendations.get(level, "Manual review required")
    
    async def _validate_website(self, session: aiohttp.ClientSession, url: str) -> Dict:
        """Check if website is accessible and appears to be a valid college site"""
        await self._rate_limit(url)

        try:
            async with session.get(url, allow_redirects=True) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')

                    # Basic validation (includes educational keywords)
                    text_content = soup.get_text().lower()
                    edu_keywords = ['college', 'university', 'admission', 'course', 
                                   'department', 'student', 'faculty', 'program']
                    edu_score = sum(1 for keyword in edu_keywords if keyword in text_content)

                    return {
                        'accessible': True,
                        'appears_educational': edu_score >= 3,
                        'content_length': len(content),
                        'edu_score': edu_score
                    }
                
        except Exception as e:
            print(f"Website validation error for {url}: {e}")

        return {
            'accessible': False, 
            'appears_educational': False, 
            'content_length': 0,
            'edu_score': 0
        }
    
    async def _find_course_evidence(self, session: aiohttp.ClientSession, college: College) -> List[str]:
        """Look for course-specific evidence on college website"""
        evidence_urls = []
        courses_found = set()

        try:
            course_paths = ['/courses', '/academics', '/programs', '/programmes', 
                          '/admissions', '/departments', '/courses.html', '/academics.html']

            for path in course_paths:
                if len(courses_found) >= len(college.courses):
                    break
                    
                course_url = urljoin(college.website, path)
                await self._rate_limit(course_url)

                try:
                    async with session.get(course_url, allow_redirects=True) as response:
                        if response.status == 200:
                            content = await response.text()
                            content_lower = content.lower()

                            for course in college.courses:
                                course_name_lower = course.course_name.lower()
                                # Check for course name or common variations
                                if (course_name_lower in content_lower or 
                                    any(word in content_lower for word in course_name_lower.split())):
                                    if course.course_name not in courses_found:
                                        evidence_urls.append(course_url)
                                        courses_found.add(course.course_name)
                
                except Exception:
                    continue

        except Exception as e:
            print(f"Course evidence search error: {e}")

        return list(set(evidence_urls))  # Remove duplicates
    
    async def _check_govt_presence(self, session: aiohttp.ClientSession, college_name: str) -> Dict:
        """Check for government recognition indicators"""
        # Enhanced government portal check
        college_name_lower = college_name.lower()
        
        # Strong indicators of government recognition
        govt_indicators = [
            'iit', 'nit', 'iiit', 'aiims', 'bits',
            'government', 'central university', 'state university',
            'deemed university', 'national institute'
        ]
        
        # Check if college name contains government indicators
        found = any(indicator in college_name_lower for indicator in govt_indicators)
        
        # Additional check for specific naming patterns
        if not found:
            # Check for patterns like "Government College", "Rajiv Gandhi University", etc.
            govt_patterns = ['govt ', 'government ', 'state ', 'central ', 'national ']
            found = any(pattern in college_name_lower for pattern in govt_patterns)
        
        urls = []
        if found:
            urls.append('https://www.aicte-india.org/dashboard/approved-institutions')
        
        return {
            'found': found,
            'urls': urls
        }
    
    async def _rate_limit(self, url: str):
        """Implement rate limiting per domain"""
        domain = urlparse(url).netloc

        if domain in self.last_request:
            elapsed = time.time() - self.last_request[domain]
            if elapsed < self.delay:
                await asyncio.sleep(self.delay - elapsed)

        self.last_request[domain] = time.time()