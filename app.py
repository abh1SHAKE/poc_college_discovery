import streamlit as st
import os
import json
import csv
import io
import asyncio
from dotenv import load_dotenv
from src.main import CollegeDiscoveryApp
from src.engines.llm_engine import CollegeDiscoveryEngine
from src.engines.validation_engine import EvidenceValidator
from src.models.college import EvidenceStatus

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

poc = CollegeDiscoveryApp(api_key=api_key, model=model)
engine = CollegeDiscoveryEngine(api_key=api_key, model=model)
validator = EvidenceValidator(delay=1.5)

st.set_page_config(page_title="College Discovery App", page_icon="🎓", layout="wide")

st.title("🎓 College Discovery App")
st.markdown("Discover colleges with AI-powered validation")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    enable_validation = st.checkbox("Enable Validation", value=True, 
                                   help="Validate colleges against websites and government databases")
    validation_delay = st.slider("Validation Delay (seconds)", 0.5, 5.0, 1.5, 0.5,
                                help="Delay between validation requests to avoid rate limiting")
    
    st.markdown("---")
    st.markdown("### Confidence Levels")
    st.markdown("""
    - **HIGH** (0.8-1.0): Auto-approve eligible
    - **MEDIUM** (0.6-0.8): Standard review
    - **LOW** (0.4-0.6): Detailed review
    - **VERY LOW** (0.0-0.4): Investigation needed
    """)
    
    st.markdown("---")
    st.markdown("### Validation Steps")
    st.markdown("""
    1. **Website Check**: Accessibility & educational content
    2. **Course Evidence**: Course details on website
    3. **Govt Verification**: Recognition indicators
    4. **Domain Quality**: Educational domain (.edu.in, .ac.in)
    """)

# Main input section
col1, col2 = st.columns(2)
with col1:
    location = st.text_input("📍 Location (city/state):", placeholder="e.g., Bangalore, Karnataka")
with col2:
    career_path = st.text_input("💼 Career Path:", placeholder="e.g., Data Science, Mechanical Engineering")

# Initialize session state
if "prompt_text_area" not in st.session_state:
    st.session_state["prompt_text_area"] = ""

# Generate prompt button
if st.button("Generate Prompt", type="secondary"):
    if not location or not career_path:
        st.error("Please provide both location and career path.")
    else:
        default_prompt = engine.create_discovery_prompt(location, career_path)
        st.session_state["prompt_text_area"] = default_prompt
        st.success("✅ Prompt generated! You can edit it below 👇")

# Prompt editing area
st.text_area(
    "Edit or refine your discovery prompt before running:",
    key="prompt_text_area",
    height=300,
    help="Modify the prompt to refine your search criteria"
)

# Run discovery button
if st.button("🔍 Run Discovery", type="primary"):
    prompt_text = st.session_state["prompt_text_area"]
    
    if not api_key:
        st.error("❌ No API key found. Please set the GROQ_API_KEY environment variable.")
    elif not prompt_text.strip():
        st.error("❌ Please generate and/or provide a valid prompt first.")
    elif not location or not career_path:
        st.error("❌ Please fill in both location and career path.")
    else:
        # Discovery phase
        with st.spinner("🔍 Discovering colleges..."):
            try:
                import re
                response = engine.client.chat.completions.create(
                    model=engine.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise educational data expert. Always return valid JSON with accurate information about Indian colleges and universities."
                        },
                        {"role": "user", "content": prompt_text}
                    ],
                    max_tokens=4000,
                    temperature=0.1,
                    top_p=0.9
                )

                content = response.choices[0].message.content.strip()
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                
                if not json_match:
                    st.error("⚠️ The model did not return valid JSON.")
                    st.stop()

                data = json.loads(json_match.group())
                colleges = engine._parse_colleges(data, location, career_path)
                
                st.success(f"✅ Found {len(colleges)} colleges!")
                
                # Validation phase
                if enable_validation and colleges:
                    st.markdown("---")
                    validator.delay = validation_delay
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with st.spinner("🔐 Validating colleges..."):
                        try:
                            # Run async validation
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            
                            async def validate_with_progress():
                                validated = []
                                for i, college in enumerate(colleges):
                                    status_text.text(f"Validating: {college.name} ({i+1}/{len(colleges)})")
                                    progress_bar.progress((i + 1) / len(colleges))
                                    
                                    # Validate single college
                                    result = await validator.validate_colleges([college])
                                    validated.extend(result)
                                    
                                return validated
                            
                            colleges = loop.run_until_complete(validate_with_progress())
                            loop.close()
                            
                            progress_bar.empty()
                            status_text.empty()
                            st.success("✅ Validation completed!")
                            
                        except Exception as e:
                            st.warning(f"⚠️ Validation encountered issues: {e}")
                            st.info("Proceeding with unvalidated data...")
                            import traceback
                            with st.expander("See validation error details"):
                                st.code(traceback.format_exc())
                
                # Store in session state
                st.session_state["colleges"] = colleges
                st.session_state["prompt_used"] = prompt_text
                st.session_state["location"] = location
                st.session_state["career_path"] = career_path
                st.session_state["validation_enabled"] = enable_validation

            except Exception as e:
                st.error(f"❌ Error during discovery: {e}")
                import traceback
                with st.expander("See error details"):
                    st.code(traceback.format_exc())

# Display results
if "colleges" in st.session_state:
    colleges = st.session_state["colleges"]
    
    st.markdown("---")
    st.header("📊 Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Colleges", len(colleges))
    with col2:
        high_conf = sum(1 for c in colleges if c.overall_confidence >= 0.8)
        st.metric("High Confidence", high_conf, 
                 delta=f"{high_conf/len(colleges)*100:.0f}%" if colleges else "0%")
    with col3:
        # Count validated colleges (either verified or partially_verified)
        validated = sum(1 for c in colleges 
                       if c.evidence_status in [EvidenceStatus.VERIFIED, EvidenceStatus.PARTIALLY_VERIFIED])
        st.metric("Validated", validated)
    with col4:
        avg_conf = sum(c.overall_confidence for c in colleges) / len(colleges) if colleges else 0
        st.metric("Avg Confidence", f"{avg_conf:.2f}")
    
    # Confidence level breakdown
    st.subheader("📈 Confidence Distribution")
    conf_levels = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "VERY_LOW": 0}
    for college in colleges:
        level = validator.get_confidence_level(college.overall_confidence)
        conf_levels[level] += 1
    
    col1, col2, col3, col4 = st.columns(4)
    colors = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🟠", "VERY_LOW": "🔴"}
    for col, (level, count) in zip([col1, col2, col3, col4], conf_levels.items()):
        with col:
            st.markdown(f"**{colors[level]} {level}**")
            st.markdown(f"{count} colleges ({count/len(colleges)*100:.0f}%)" if colleges else "0 colleges")
    
    # Display colleges
    st.markdown("---")
    st.subheader("🏫 College Details")
    
    for i, college in enumerate(colleges):
        confidence_level = validator.get_confidence_level(college.overall_confidence)
        
        # Evidence status display
        status_display = {
            EvidenceStatus.VERIFIED: "✅ Verified",
            EvidenceStatus.PARTIALLY_VERIFIED: "⚠️ Partially Verified",
            EvidenceStatus.NO_EVIDENCE_FOUND: "❌ No Evidence Found"
        }
        evidence_display = status_display.get(college.evidence_status, 
                                              college.evidence_status.value if hasattr(college.evidence_status, 'value') else str(college.evidence_status))
        
        with st.expander(
            f"**{college.name}** - {confidence_level} "
            f"(Confidence: {college.overall_confidence:.2f}) - {evidence_display}"
        ):
            # Basic Info
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**📍 Location:** {college.city}, {college.state}")
                st.markdown(f"**🏢 Type:** {college.type}")
                st.markdown(f"**🌐 Website:** [{college.website}]({college.website})")
            
            with col2:
                action = validator.get_action_recommendation(college.overall_confidence)
                st.info(f"**Recommended Action:**\n{action}")
            
            # Validation Details
            if hasattr(college, 'validation_details') and college.validation_details:
                st.markdown("---")
                st.markdown("### 🔍 Validation Results")
                
                details = college.validation_details
                
                # Create 4 columns for validation steps
                v_col1, v_col2, v_col3, v_col4 = st.columns(4)
                
                with v_col1:
                    st.markdown("**1️⃣ Website Check**")
                    if details.get('website_accessible'):
                        st.success("✅ Accessible")
                        if details.get('website_appears_educational'):
                            st.caption(f"Educational keywords: {details.get('edu_keywords_found', 0)}/8")
                        else:
                            st.warning("⚠️ Not educational")
                    else:
                        st.error("❌ Not accessible")
                    
                    adj = details.get('adjustments', {}).get('website', 0)
                    st.caption(f"Adjustment: {adj:+.2f}")
                
                with v_col2:
                    st.markdown("**2️⃣ Course Evidence**")
                    courses_found = details.get('courses_found', 0)
                    total_courses = details.get('total_courses', 0)
                    
                    if courses_found > 0:
                        match_pct = details.get('course_match_percentage', 0)
                        st.success(f"✅ {courses_found}/{total_courses} courses")
                        st.caption(f"Match: {match_pct:.0f}%")
                    else:
                        st.error(f"❌ 0/{total_courses} found")
                    
                    adj = details.get('adjustments', {}).get('course_evidence', 0)
                    st.caption(f"Adjustment: {adj:+.2f}")
                
                with v_col3:
                    st.markdown("**3️⃣ Govt Verification**")
                    if details.get('govt_verified'):
                        st.success("✅ Verified")
                        st.caption("Govt indicators found")
                    else:
                        st.info("ℹ️ Not verified")
                        st.caption("No govt indicators")
                    
                    adj = details.get('adjustments', {}).get('govt_verification', 0)
                    st.caption(f"Adjustment: {adj:+.2f}")
                
                with v_col4:
                    st.markdown("**4️⃣ Domain Quality**")
                    domain_type = details.get('domain_type', 'Unknown')
                    adj = details.get('adjustments', {}).get('domain_quality', 0)
                    
                    if adj > 0:
                        st.success(f"✅ {domain_type}")
                    else:
                        st.info(f"ℹ️ {domain_type}")
                    
                    st.caption(f"Adjustment: {adj:+.2f}")
            
            # Evidence URLs
            if college.evidence_urls and len(college.evidence_urls) > 0:
                st.markdown("---")
                st.markdown("**🔗 Evidence URLs:**")
                for url in college.evidence_urls[:5]:  # Show first 5
                    st.markdown(f"- [{url}]({url})")
            
            # Courses
            if college.courses and len(college.courses) > 0:
                st.markdown("---")
                st.markdown(f"**📚 Courses ({len(college.courses)}):**")
                for course in college.courses:
                    st.markdown(f"- **{course.name}** ({course.degree_level}) - {course.duration}")
                    if course.annual_fees:
                        st.markdown(f"  💰 Fees: {course.annual_fees:}/year")
                    if course.entrance_exams and len(course.entrance_exams) > 0:
                        st.markdown(f"  📝 Exams: {', '.join(course.entrance_exams)}")
                    if course.specializations and len(course.specializations) > 0:
                        st.markdown(f"  🎯 Specializations: {', '.join(course.specializations)}")
    
    # Download section
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    # Status display mapping
    status_display = {
        EvidenceStatus.VERIFIED: "Verified",
        EvidenceStatus.PARTIALLY_VERIFIED: "Partially Verified",
        EvidenceStatus.NO_EVIDENCE_FOUND: "No Evidence Found"
    }
    
    # JSON download
    with col1:
        json_data = {
            "metadata": {
                "location": st.session_state.get("location", ""),
                "career_path": st.session_state.get("career_path", ""),
                "total_colleges": len(colleges),
                "validation_enabled": st.session_state.get("validation_enabled", False),
                "prompt_used": st.session_state.get("prompt_used", "")
            },
            "colleges": [
                {
                    "name": c.name,
                    "city": c.city,
                    "state": c.state,
                    "type": c.type,
                    "website": c.website,
                    "confidence": c.overall_confidence,
                    "confidence_level": validator.get_confidence_level(c.overall_confidence),
                    "evidence_status": status_display.get(c.evidence_status, 
                                                          c.evidence_status.value if hasattr(c.evidence_status, 'value') else str(c.evidence_status)),
                    "evidence_urls": c.evidence_urls if c.evidence_urls else [],
                    "recommended_action": validator.get_action_recommendation(c.overall_confidence),
                    "validation_details": c.validation_details if hasattr(c, 'validation_details') else {},
                    "courses": [
                        {
                            "name": course.name,
                            "degree": course.degree_level,
                            "duration": course.duration,
                            "fees": course.annual_fees,
                            "seats": course.seats,
                            "exams": course.entrance_exams if course.entrance_exams else [],
                            "specializations": course.specializations if course.specializations else []
                        }
                        for course in c.courses
                    ] if c.courses else []
                }
                for c in colleges
            ]
        }
        
        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
        location_safe = st.session_state.get("location", "").replace(' ', '_').replace(',', '')
        career_safe = st.session_state.get("career_path", "").replace(' ', '_')
        
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"colleges_{location_safe}_{career_safe}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # CSV download
    with col2:
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow([
            'College Name', 'City', 'State', 'Type', 'Website',
            'Confidence Score', 'Confidence Level', 'Evidence Status',
            'Recommended Action', 'Website Accessible', 'Courses Found',
            'Govt Verified', 'Domain Type', 'Evidence URLs',
            'Course Name', 'Degree Level', 'Duration', 'Annual Fees',
            'Seats', 'Entrance Exams', 'Specializations'
        ])
        
        for college in colleges:
            evidence_display = status_display.get(college.evidence_status, 
                                                  college.evidence_status.value if hasattr(college.evidence_status, 'value') else str(college.evidence_status))
            
            # Get validation details
            val_details = college.validation_details if hasattr(college, 'validation_details') else {}
            
            base_row = [
                college.name, college.city, college.state,
                college.type, college.website,
                f"{college.overall_confidence:.2f}",
                validator.get_confidence_level(college.overall_confidence),
                evidence_display,
                validator.get_action_recommendation(college.overall_confidence),
                "Yes" if val_details.get('website_accessible') else "No",
                f"{val_details.get('courses_found', 0)}/{val_details.get('total_courses', 0)}",
                "Yes" if val_details.get('govt_verified') else "No",
                val_details.get('domain_type', 'Unknown'),
                "; ".join(college.evidence_urls) if college.evidence_urls else ""
            ]
            
            if college.courses and len(college.courses) > 0:
                for course in college.courses:
                    writer.writerow(base_row + [
                        course.name, course.degree_level,
                        course.duration, course.annual_fees or "",
                        course.seats or "",
                        "; ".join(course.entrance_exams) if course.entrance_exams else "",
                        "; ".join(course.specializations) if course.specializations else ""
                    ])
            else:
                writer.writerow(base_row + ["", "", "", "", "", "", ""])
        
        csv_str = output.getvalue()
        output.close()
        
        st.download_button(
            label="📥 Download CSV",
            data=csv_str,
            file_name=f"colleges_{location_safe}_{career_safe}.csv",
            mime="text/csv",
            use_container_width=True
        )