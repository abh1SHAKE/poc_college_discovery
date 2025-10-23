import streamlit as st
import os
import json
import csv
import io
import asyncio
import re
from dotenv import load_dotenv
from src.engines.llm_engine import CollegeDiscoveryEngine
from src.engines.validation_engine import EvidenceValidator
from src.models.college import EvidenceStatus

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

engine = CollegeDiscoveryEngine(api_key=api_key, model=model)
validator = EvidenceValidator(delay=1.5)

st.set_page_config(page_title="College Discovery App", page_icon="🎓", layout="wide")

st.title("🎓 College Discovery App")
st.markdown("Discover colleges with AI-powered validation using a two-step discovery process")

with st.sidebar:
    st.header("⚙️ Settings")
    enable_validation = st.checkbox("Enable Validation", value=True, 
                                   help="Validate colleges against websites and government databases")
    validation_delay = st.slider("Validation Delay (seconds)", 0.5, 5.0, 1.5, 0.5,
                                help="Delay between validation requests to avoid rate limiting")
    
    st.markdown("---")
    st.markdown("### Discovery Process")
    st.markdown("""
    **Step 1:** Find all colleges in location (40-60+)
    
    **Step 2:** For each college, discover all courses
    
    **Step 3:** Validate each college (if enabled)
    """)
    
    st.markdown("---")
    st.markdown("### Confidence Levels")
    st.markdown("""
    - **HIGH** (0.8-1.0): Auto-approve eligible
    - **MEDIUM** (0.6-0.8): Standard review
    - **LOW** (0.4-0.6): Detailed review
    - **VERY LOW** (0.0-0.4): Investigation needed
    """)

# Main input section
col1, col2 = st.columns(2)
with col1:
    location = st.text_input("📍 Location (city/state):", placeholder="e.g., Bangalore, Karnataka")
with col2:
    career_path = st.text_input("💼 Career Path (Optional - filters results):", 
                               placeholder="e.g., Data Science, Engineering")

st.info("💡 **Tip:** Leave Career Path empty to discover all colleges and courses in the location, or specify to filter results.")

# Initialize session state for prompts
if "college_prompt" not in st.session_state:
    st.session_state["college_prompt"] = ""
if "course_prompt_template" not in st.session_state:
    st.session_state["course_prompt_template"] = ""

# Generate prompts button
if st.button("Generate Prompts", type="secondary"):
    if not location:
        st.error("Please provide a location first.")
    else:
        # Generate college discovery prompt
        college_prompt = engine.create_college_list_prompt(location)
        st.session_state["college_prompt"] = college_prompt
        
        # Generate course discovery prompt template (will be customized per college)
        course_prompt = engine.create_course_discovery_prompt(
            "{COLLEGE_NAME}",
            "{COLLEGE_WEBSITE}",
            career_path if career_path else None
        )
        st.session_state["course_prompt_template"] = course_prompt
        
        st.success("✅ Prompts generated! You can edit them below before running discovery 👇")

# Display and allow editing of prompts
if st.session_state["college_prompt"]:
    st.markdown("---")
    st.subheader("📝 Prompt Configuration")
    
    with st.expander("🏫 College Discovery Prompt (Step 1)", expanded=True):
        st.markdown("This prompt will be used to discover all colleges in the location:")
        edited_college_prompt = st.text_area(
            "Edit College Discovery Prompt:",
            value=st.session_state["college_prompt"],
            height=300,
            key="college_prompt_editor"
        )
        st.session_state["college_prompt"] = edited_college_prompt
    
    with st.expander("📚 Course Discovery Prompt Template (Step 2)", expanded=False):
        st.markdown("This prompt template will be used for each college (variables `{COLLEGE_NAME}` and `{COLLEGE_WEBSITE}` will be replaced):")
        edited_course_prompt = st.text_area(
            "Edit Course Discovery Prompt Template:",
            value=st.session_state["course_prompt_template"],
            height=300,
            key="course_prompt_editor"
        )
        st.session_state["course_prompt_template"] = edited_course_prompt

# Run discovery button
if st.button("🔍 Run Discovery", type="primary"):
    
    if not api_key:
        st.error("❌ No API key found. Please set the GROQ_API_KEY environment variable.")
    elif not location:
        st.error("❌ Please provide a location.")
    elif not st.session_state["college_prompt"]:
        st.error("❌ Please generate prompts first by clicking 'Generate Prompts'.")
    else:
        # Progress tracking containers
        step1_container = st.container()
        step2_container = st.container()
        step3_container = st.container()
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Step 1: Discover Colleges using custom prompt
            with step1_container:
                st.markdown("---")
                st.subheader("Step 1: Discovering Colleges")
                step1_status = st.empty()
                step1_status.text(f"🔍 Searching for colleges in {location}...")
            
            # Custom implementation using the edited prompt
            async def discover_colleges_with_custom_prompt():
                try:
                    response = engine.client.chat.completions.create(
                        model=engine.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a precise educational data expert. Always return valid JSON with accurate information about Indian colleges and universities."
                            },
                            {"role": "user", "content": st.session_state["college_prompt"]}
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
                    return engine._parse_colleges_basic(data, location)

                except Exception as e:
                    print(f"Error in college list discovery: {e}")
                    return []
            
            colleges = loop.run_until_complete(discover_colleges_with_custom_prompt())
            
            if not colleges:
                step1_status.error("❌ No colleges found. Please try a different location or modify the prompt.")
                st.stop()
            
            step1_status.success(f"✅ Found {len(colleges)} colleges!")
            
            # Step 2: Discover Courses using custom prompt template
            with step2_container:
                st.markdown("---")
                st.subheader("Step 2: Discovering Courses")
                step2_progress = st.progress(0)
                step2_status = st.empty()
                
                async def discover_all_courses_with_custom_prompt():
                    colleges_with_courses = []
                    total = len(colleges)
                    
                    for idx, college in enumerate(colleges):
                        step2_status.text(f"📚 Discovering courses: {college.name} ({idx+1}/{total})")
                        step2_progress.progress((idx + 1) / total)
                        
                        # Replace placeholders in template
                        custom_course_prompt = st.session_state["course_prompt_template"].replace(
                            "{COLLEGE_NAME}", college.name
                        ).replace(
                            "{COLLEGE_WEBSITE}", college.website
                        )
                        
                        try:
                            response = engine.client.chat.completions.create(
                                model=engine.model,
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "You are a precise educational data expert. Always return valid JSON with accurate course information."
                                    },
                                    {"role": "user", "content": custom_course_prompt}
                                ],
                                max_tokens=3000,
                                temperature=0.1,
                                top_p=0.9
                            )

                            content = response.choices[0].message.content.strip()
                            json_match = re.search(r'\{.*\}', content, re.DOTALL)
                            
                            if json_match:
                                data = json.loads(json_match.group())
                                courses = engine._parse_courses(data, college.website)
                                college.courses = courses
                            else:
                                college.courses = []
                                
                        except Exception as e:
                            print(f"Error discovering courses for {college.name}: {e}")
                            college.courses = []
                        
                        colleges_with_courses.append(college)
                    
                    return colleges_with_courses
                
                colleges = loop.run_until_complete(discover_all_courses_with_custom_prompt())
            
            total_courses = sum(len(c.courses) for c in colleges)
            step2_status.success(f"✅ Discovered {total_courses} courses across {len(colleges)} colleges!")
            
            # Filter by career path if specified
            if career_path:
                original_count = len(colleges)
                colleges = [c for c in colleges if len(c.courses) > 0]
                if len(colleges) < original_count:
                    st.info(f"ℹ️ Filtered to {len(colleges)} colleges with {career_path}-related courses")
            
            # Step 3: Validation
            if enable_validation and colleges:
                with step3_container:
                    st.markdown("---")
                    st.subheader("Step 3: Validating Colleges")
                    
                    validator.delay = validation_delay
                    val_progress = st.progress(0)
                    val_status = st.empty()
                    
                    async def validate_with_progress():
                        validated = []
                        for i, college in enumerate(colleges):
                            val_status.text(f"🔐 Validating: {college.name} ({i+1}/{len(colleges)})")
                            val_progress.progress((i + 1) / len(colleges))
                            
                            result = await validator.validate_colleges([college])
                            validated.extend(result)
                        
                        return validated
                    
                    try:
                        colleges = loop.run_until_complete(validate_with_progress())
                        val_status.success("✅ Validation completed!")
                    except Exception as e:
                        st.warning(f"⚠️ Validation encountered issues: {e}")
                        st.info("Proceeding with unvalidated data...")
            
            loop.close()
            
            # Store in session state
            st.session_state["colleges"] = colleges
            st.session_state["location"] = location
            st.session_state["career_path"] = career_path or "All Programs"
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
    total_courses = sum(len(c.courses) for c in colleges)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Colleges", len(colleges))
    with col2:
        st.metric("Total Courses", total_courses)
    with col3:
        high_conf = sum(1 for c in colleges if c.overall_confidence >= 0.8)
        st.metric("High Confidence", high_conf, 
                 delta=f"{high_conf/len(colleges)*100:.0f}%" if colleges else "0%")
    with col4:
        validated = sum(1 for c in colleges 
                       if c.evidence_status in [EvidenceStatus.VERIFIED, EvidenceStatus.PARTIALLY_VERIFIED])
        st.metric("Validated", validated)
    
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
    
    # Filter options
    st.markdown("---")
    st.subheader("🔍 Filter Results")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        conf_filter = st.selectbox(
            "Confidence Level",
            ["All", "HIGH", "MEDIUM", "LOW", "VERY_LOW"]
        )
    
    with filter_col2:
        type_options = ["All"] + list(set(c.type for c in colleges if c.type))
        type_filter = st.selectbox("College Type", type_options)
    
    with filter_col3:
        evidence_options = ["All", "Verified", "Partially Verified", "No Evidence"]
        evidence_filter = st.selectbox("Evidence Status", evidence_options)
    
    # Apply filters
    filtered_colleges = colleges
    
    if conf_filter != "All":
        filtered_colleges = [c for c in filtered_colleges 
                           if validator.get_confidence_level(c.overall_confidence) == conf_filter]
    
    if type_filter != "All":
        filtered_colleges = [c for c in filtered_colleges if c.type == type_filter]
    
    if evidence_filter != "All":
        status_map = {
            "Verified": EvidenceStatus.VERIFIED,
            "Partially Verified": EvidenceStatus.PARTIALLY_VERIFIED,
            "No Evidence": EvidenceStatus.NO_EVIDENCE_FOUND
        }
        filtered_colleges = [c for c in filtered_colleges 
                           if c.evidence_status == status_map[evidence_filter]]
    
    st.info(f"Showing {len(filtered_colleges)} of {len(colleges)} colleges")
    
    # Display colleges
    st.markdown("---")
    st.subheader("🏫 College Details")
    
    # Status display mapping
    status_display = {
        EvidenceStatus.VERIFIED: "✅ Verified",
        EvidenceStatus.PARTIALLY_VERIFIED: "⚠️ Partially Verified",
        EvidenceStatus.NO_EVIDENCE_FOUND: "❌ No Evidence Found"
    }
    
    for i, college in enumerate(filtered_colleges):
        confidence_level = validator.get_confidence_level(college.overall_confidence)
        evidence_display = status_display.get(college.evidence_status, 
                                              college.evidence_status.value if hasattr(college.evidence_status, 'value') else str(college.evidence_status))
        
        with st.expander(
            f"**{college.name}** - {confidence_level} "
            f"(Confidence: {college.overall_confidence:.2f}) - {evidence_display} - "
            f"{len(college.courses)} courses"
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
                
                v_col1, v_col2, v_col3, v_col4 = st.columns(4)
                
                with v_col1:
                    st.markdown("**1️⃣ Website Check**")
                    if details.get('website_accessible'):
                        st.success("✅ Accessible")
                        if details.get('website_appears_educational'):
                            st.caption(f"Educational keywords: {details.get('edu_keywords_found', 0)}/8")
                    else:
                        st.error("❌ Not accessible")
                    
                    adj = details.get('adjustments', {}).get('website', 0)
                    st.caption(f"Adjustment: {adj:+.2f}")
                
                with v_col2:
                    st.markdown("**2️⃣ Course Evidence**")
                    courses_found = details.get('courses_found', 0)
                    total_courses_val = details.get('total_courses', 0)
                    
                    if courses_found > 0:
                        match_pct = details.get('course_match_percentage', 0)
                        st.success(f"✅ {courses_found}/{total_courses_val} courses")
                        st.caption(f"Match: {match_pct:.0f}%")
                    else:
                        st.error(f"❌ 0/{total_courses_val} found")
                    
                    adj = details.get('adjustments', {}).get('course_evidence', 0)
                    st.caption(f"Adjustment: {adj:+.2f}")
                
                with v_col3:
                    st.markdown("**3️⃣ Govt Verification**")
                    if details.get('govt_verified'):
                        st.success("✅ Verified")
                    else:
                        st.info("ℹ️ Not verified")
                    
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
                for url in college.evidence_urls[:5]:
                    st.markdown(f"- [{url}]({url})")
            
            # Courses
            if college.courses and len(college.courses) > 0:
                st.markdown("---")
                st.markdown(f"**📚 Courses ({len(college.courses)}):**")
                
                # Group courses by degree level
                courses_by_level = {}
                for course in college.courses:
                    level = course.degree_level
                    if level not in courses_by_level:
                        courses_by_level[level] = []
                    courses_by_level[level].append(course)
                
                for level, level_courses in courses_by_level.items():
                    st.markdown(f"**{level} Programs ({len(level_courses)}):**")
                    for course in level_courses:
                        st.markdown(f"- **{course.name}** - {course.duration}")
                        details_list = []
                        if course.annual_fees:
                            details_list.append(f"💰 {course.annual_fees}/year")
                        if course.seats:
                            details_list.append(f"🪑 {course.seats} seats")
                        if course.entrance_exams and len(course.entrance_exams) > 0:
                            details_list.append(f"📝 {', '.join(course.entrance_exams)}")
                        if details_list:
                            st.markdown(f"  {' | '.join(details_list)}")
                        if course.specializations and len(course.specializations) > 0:
                            st.markdown(f"  🎯 Specializations: {', '.join(course.specializations)}")
    
    # Download section
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    # JSON download
    with col1:
        json_data = {
            "metadata": {
                "location": st.session_state.get("location", ""),
                "career_path": st.session_state.get("career_path", ""),
                "total_colleges": len(colleges),
                "total_courses": total_courses,
                "validation_enabled": st.session_state.get("validation_enabled", False)
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
                    "evidence_status": status_display.get(c.evidence_status, str(c.evidence_status)),
                    "evidence_urls": c.evidence_urls if c.evidence_urls else [],
                    "total_courses": len(c.courses),
                    "courses": [
                        {
                            "name": course.name,
                            "degree_level": course.degree_level,
                            "duration": course.duration,
                            "annual_fees": course.annual_fees,
                            "seats": course.seats,
                            "entrance_exams": course.entrance_exams if course.entrance_exams else [],
                            "specializations": course.specializations if course.specializations else []
                        }
                        for course in c.courses
                    ]
                }
                for c in colleges
            ]
        }
        
        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
        location_safe = st.session_state.get("location", "").replace(' ', '_').replace(',', '')
        
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"colleges_{location_safe}.json",
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
            'Recommended Action', 'Total Courses', 'Website Accessible',
            'Courses Found', 'Govt Verified', 'Domain Type',
            'Course Name', 'Degree Level', 'Duration', 'Annual Fees',
            'Seats', 'Entrance Exams', 'Specializations'
        ])
        
        for college in colleges:
            evidence_display_csv = status_display.get(college.evidence_status, str(college.evidence_status))
            
            val_details = college.validation_details if hasattr(college, 'validation_details') else {}
            
            base_row = [
                college.name, college.city, college.state,
                college.type, college.website,
                f"{college.overall_confidence:.2f}",
                validator.get_confidence_level(college.overall_confidence),
                evidence_display_csv,
                validator.get_action_recommendation(college.overall_confidence),
                len(college.courses),
                "Yes" if val_details.get('website_accessible') else "No",
                f"{val_details.get('courses_found', 0)}/{val_details.get('total_courses', 0)}",
                "Yes" if val_details.get('govt_verified') else "No",
                val_details.get('domain_type', 'Unknown')
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
            file_name=f"colleges_{location_safe}.csv",
            mime="text/csv",
            use_container_width=True
        )