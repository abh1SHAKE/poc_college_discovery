import asyncio
import json
import csv
import os
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

from src.engines.llm_engine import CollegeDiscoveryEngine
from src.engines.validation_engine import EvidenceValidator
from src.models.college import College

class CollegeDiscoveryApp:
    def __init__(self, api_key: str, model: str = None):
        """Initialize discovery app with Groq API key and model"""
        self.discovery_engine = CollegeDiscoveryEngine(api_key, model=model)
        self.validator = EvidenceValidator(delay=1.5)

    async def run_discovery(self, location: str, career_path: str) -> Dict:
        """Run complete college discovery pipeline"""
        print(f"Starting discovery for {location} - {career_path}")
        
        # Step 1: LLM Discovery
        print("Phase 1: Discovering colleges with LLM...")
        colleges = await self.discovery_engine.discover_colleges(location, career_path)
        print(f"Found {len(colleges)} colleges")
        
        if not colleges:
            print("No colleges found by LLM")
            return self._generate_results(location, career_path, [])
        
        # Step 2: Evidence Validation
        # print("Phase 2: Validating evidence...")
        # validated_colleges = await self.validator.validate_colleges(colleges)

        # Skipping validation for now
        validated_colleges = colleges
        
        # Step 3: Generate results
        results = self._generate_results(location, career_path, validated_colleges)
        return results

    def _generate_results(self, location: str, career_path: str, colleges: List[College]) -> Dict:
        """Generate structured results"""
        verified_count = sum(1 for c in colleges if c.evidence_status.value == "Verified")
        total_courses = sum(len(c.courses) for c in colleges)
        avg_confidence = sum(c.overall_confidence for c in colleges) / len(colleges) if colleges else 0
        
        return {
            "search_query": {
                "location": location,
                "career_path": career_path,
                "timestamp": datetime.now().isoformat()
            },
            "summary": {
                "total_colleges": len(colleges),
                "verified_colleges": verified_count,
                "total_courses": total_courses,
                "avg_confidence": round(avg_confidence, 2)
            },
            "colleges": [self._college_to_dict(college) for college in colleges]
        }

    def _college_to_dict(self, college: College) -> Dict:
        """Convert college object to dictionary"""
        return {
            "name": college.name,
            "city": college.city,
            "state": college.state,
            "type": college.type,
            "website": college.website,
            "overall_confidence": college.overall_confidence,
            "last_collected": college.last_collected.isoformat(),
            "verification_status": college.verification_status.value,
            "evidence_status": college.evidence_status.value,
            "evidence_urls": college.evidence_urls,
            "courses": [
                {
                    "course_name": course.name,
                    "degree_level": course.degree_level,
                    "official_source_url": course.official_source_url,
                    "row_confidence": course.row_confidence,
                    "duration": course.duration,
                    "annual_fees": course.annual_fees,
                    "seats": course.seats,
                    "entrance_exams": course.entrance_exams,
                    "specializations": course.specializations,
                    "evidence_urls": course.evidence_urls
                }
                for course in college.courses
            ]
        }

    def save_results(self, results: Dict, output_format: str = "both"):
        """Save results to JSON and/or CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        location_clean = results["search_query"]["location"].replace(", ", "_").replace(" ", "_")
        career_clean = results["search_query"]["career_path"].replace(" ", "_")
        
        base_filename = f"{location_clean}_{career_clean}_{timestamp}"
        
        if output_format in ["json", "both"]:
            json_filename = f"outputs/{base_filename}.json"
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {json_filename}")
        
        if output_format in ["csv", "both"]:
            csv_filename = f"outputs/{base_filename}.csv"
            self._save_csv(results, csv_filename)
            print(f"Results saved to {csv_filename}")

    def _save_csv(self, results: Dict, filename: str):
        """Save results to CSV format"""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Headers
            writer.writerow([
                'College Name', 'City', 'State', 'Type', 'Website',
                'Overall Confidence', 'Verification Status', 'Evidence Status',
                'Course Name', 'Degree Level', 'Duration', 'Annual Fees',
                'Seats', 'Entrance Exams', 'Specializations'
            ])
            
            # Data rows
            for college in results["colleges"]:
                base_row = [
                    college["name"], college["city"], college["state"],
                    college["type"], college["website"], college["overall_confidence"],
                    college["verification_status"], college["evidence_status"]
                ]
                
                if college["courses"]:
                    for course in college["courses"]:
                        row = base_row + [
                            course["course_name"], course["degree_level"],
                            course["duration"], course["annual_fees"], course["seats"],
                            "; ".join(course["entrance_exams"]) if course["entrance_exams"] else "",
                            "; ".join(course["specializations"]) if course["specializations"] else ""
                        ]
                        writer.writerow(row)
                else:
                    # College with no courses
                    row = base_row + ["", "", "", "", "", "", ""]
                    writer.writerow(row)

# CLI Interface
async def main():
    """Main test function"""
    load_dotenv()
    
    # Get Groq API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Please set GROQ_API_KEY in your .env file")
        return
    
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Initialize POC
    poc = CollegeDiscoveryApp(api_key, model=model)
    
    # Example queries for testing
    test_queries = [
        ("Bangalore, Karnataka", "Computer Science Engineering"),
        ("Mumbai, Maharashtra", "Mechanical Engineering"),
        ("Delhi", "Data Science"),
    ]
    
    print("College Discovery POC - Running Test Queries")
    print("=" * 50)
    
    for location, career_path in test_queries:
        try:
            print(f"\nProcessing: {location} - {career_path}")
            results = await poc.run_discovery(location, career_path)
            
            # Save results
            poc.save_results(results, "both")
            
            # Print summary
            summary = results["summary"]
            print(f"Summary: {summary['total_colleges']} colleges, "
                  f"{summary['verified_colleges']} verified, "
                  f"avg confidence: {summary['avg_confidence']}")
                  
        except Exception as e:
            print(f"Error processing {location} - {career_path}: {e}")
        
        print("-" * 30)

def interactive_main():
    """Interactive version for custom queries"""
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Please set GROQ_API_KEY in your .env file")
        return
    
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    
    os.makedirs("outputs", exist_ok=True)
    poc = CollegeDiscoveryApp(api_key, model=model)
    
    print("College Discovery POC - Interactive Mode")
    print("=" * 40)
    
    while True:
        print("\nEnter your search criteria (or 'quit' to exit):")
        location = input("Location (city/state): ").strip()
        
        if location.lower() == 'quit':
            break
        
        career_path = input("Career Path: ").strip()
        
        if not location or not career_path:
            print("Please provide both location and career path.")
            continue
        
        try:
            print(f"\nSearching for colleges in {location} offering {career_path}...")
            results = asyncio.run(poc.run_discovery(location, career_path))
            
            # Save results
            poc.save_results(results, "both")
            
            # Display results
            print(f"\n{results['summary']['total_colleges']} colleges found:")            
            for college in results["colleges"][:5]:  # Show first 5
                print(f"- {college['name']} ({college['overall_confidence']:.2f} confidence)")
                print(f"  {len(college['courses'])} courses, Status: {college['evidence_status']}")
            
            if len(results["colleges"]) > 5:
                print(f"... and {len(results['colleges']) - 5} more (check output files)")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_main()
    else:
        asyncio.run(main())
