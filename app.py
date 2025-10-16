import streamlit as st
import os
import json
import csv
import io
from dotenv import load_dotenv
from src.main import CollegeDiscoveryApp
from src.engines.llm_engine import CollegeDiscoveryEngine

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

poc = CollegeDiscoveryApp(api_key=api_key, model=model)
engine = CollegeDiscoveryEngine(api_key=api_key, model=model)

st.title("🎓 College Discovery App")

location = st.text_input("Enter location (city/state):")
career_path = st.text_input("Enter career path (e.g., Data Science, Mechanical Engineering):")

if "prompt_text_area" not in st.session_state:
    st.session_state["prompt_text_area"] = ""

if st.button("Generate Prompt"):
    if not location or not career_path:
        st.error("Please provide both location and career path.")
    else:
        default_prompt = engine.create_discovery_prompt(location, career_path)
        st.session_state["prompt_text_area"] = default_prompt
        st.success("Prompt generated! You can edit it below 👇")

st.text_area(
    "Edit or refine your discovery prompt before running:",
    key="prompt_text_area",
    height=400
)

if st.button("Run Discovery"):
    prompt_text = st.session_state["prompt_text_area"]
    if not api_key:
        st.error("No API key found. Please set the GROQ_API_KEY environment variable.")
    elif not prompt_text.strip():
        st.error("Please generate and/or provide a valid prompt first.")
    elif not location or not career_path:
        st.error("Please fill in both location and career path.")
    else:
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
                results = poc._generate_results(location, career_path, colleges)
                results["prompt_used"] = prompt_text

                st.session_state["results"] = results
                st.success("✅ Discovery completed!")

            except Exception as e:
                st.error(f"Error during discovery: {e}")

if "results" in st.session_state:
    results = st.session_state["results"]

    st.subheader("📊 Summary")
    st.json(results["summary"])

    st.subheader("🏫 Top Colleges")
    for college in results["colleges"][:5]:
        st.markdown(f"**{college['name']}** ({college['city']}, {college['state']})")
        st.caption(f"{len(college['courses'])} courses | Confidence: {college['overall_confidence']:.2f}")

    json_str = json.dumps(results, indent=2, ensure_ascii=False)
    st.download_button(
        label="💾 Download Results as JSON",
        data=json_str,
        file_name="results.json",
        mime="application/json"
    )

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        'Prompt Used', 'College Name', 'City', 'State', 'Type', 'Website',
        'Overall Confidence', 'Verification Status', 'Evidence Status',
        'Course Name', 'Degree Level', 'Duration', 'Annual Fees',
        'Seats', 'Entrance Exams', 'Specializations'
    ])

    for college in results["colleges"]:
        base_row = [
            results["prompt_used"],
            college["name"], college["city"], college["state"],
            college["type"], college["website"], college["overall_confidence"],
            college["verification_status"], college["evidence_status"]
        ]
        if college["courses"]:
            for course in college["courses"]:
                writer.writerow(base_row + [
                    course["course_name"], course["degree_level"],
                    course["duration"], course["annual_fees"], course["seats"],
                    "; ".join(course["entrance_exams"]) if course["entrance_exams"] else "",
                    "; ".join(course["specializations"]) if course["specializations"] else ""
                ])
        else:
            writer.writerow(base_row + ["", "", "", "", "", "", ""])

    csv_str = output.getvalue()
    output.close()

    st.download_button(
        label="📄 Download Results as CSV",
        data=csv_str,
        file_name="results.csv",
        mime="text/csv"
    )
