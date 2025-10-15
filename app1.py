import os
import json
import sqlite3
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import Counter
import time

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# ------------------ Config ------------------
DB_PATH = os.getenv("SYMPTOM_DB", "symptoms_history.db")
HF_TOKEN = os.getenv("HF_TOKEN", "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")
MODEL_NAME = os.getenv("HF_MODEL_NAME", "EleutherAI/gpt-neo-125M")
MAX_CONDITIONS = 5

# ------------------ Database ------------------
def init_db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symptoms TEXT NOT NULL,
            response_json TEXT NOT NULL,
            raw_text TEXT,
            created_at TEXT NOT NULL
        )
    """)
    con.commit()
    return con
DB_CON = init_db()

def save_query(symptoms: str, response_json: Dict[str, Any], raw_text: str = "") -> int:
    cur = DB_CON.cursor()
    cur.execute(
        "INSERT INTO queries (symptoms, response_json, raw_text, created_at) VALUES (?,?,?,?)",
        (symptoms, json.dumps(response_json, ensure_ascii=False), raw_text, datetime.utcnow().isoformat())
    )
    DB_CON.commit()
    return cur.lastrowid

def delete_query(query_id: int):
    cur = DB_CON.cursor()
    cur.execute("DELETE FROM queries WHERE id=?", (query_id,))
    DB_CON.commit()

def fetch_history(limit: int = 50) -> List[Dict[str, Any]]:
    cur = DB_CON.cursor()
    cur.execute("SELECT id, symptoms, response_json, raw_text, created_at FROM queries ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    out = []
    for r in rows:
        try: resp = json.loads(r[2])
        except: resp = {"_raw": r[2]}
        out.append({"id": r[0], "symptoms": r[1], "response": resp, "raw_text": r[3], "created_at": r[4]})
    return out

# ------------------ LLM & Prompt ------------------
def build_json_prompt(symptoms: str) -> str:
    prompt = textwrap.dedent(f"""
    You are a medically-informed assistant constrained to provide educational information only.
    NEVER give definitive diagnoses. Your output MUST be valid JSON with:
    {{
      "conditions": [{{"name": "...", "confidence": "low|medium|high", "notes": "..."}}],
      "recommendations": ["..."],
      "disclaimer": "short disclaimer",
      "red_flags": ["..."]
    }}
    Constraints:
    - Up to {MAX_CONDITIONS} conditions.
    - Notes <220 chars.
    - Confidence: low, medium, high.
    - Only urgent if severe.
    Symptoms:
    \"\"\"\n{symptoms}\n\"\"\"
    """).strip()
    return prompt

def extract_first_json_blob(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1: return None
    depth = 0
    for i in range(start,len(text)):
        if text[i]=="{": depth+=1
        elif text[i]=="}":
            depth-=1
            if depth==0: return text[start:i+1]
    return None

def mock_response_for_prompt(prompt: str) -> Dict[str,Any]:
    return {
        "conditions":[{"name":"Common cold","confidence":"low","notes":"Viral upper respiratory infection."},
                      {"name":"Allergic rhinitis","confidence":"low","notes":"Sneezing/runny nose."}],
        "recommendations":["Rest, hydrate, paracetamol if needed.",
                           "See clinician if fever >3 days or breathing difficulty develops.",
                           "Emergency only if chest pain, severe breathlessness, fainting."],
        "disclaimer":"Educational only. Not a diagnosis.",
        "red_flags":["chest pain","severe breathlessness","loss of consciousness"],
        "_raw":"MOCK: No LLM response."
    }

RED_FLAG_KEYWORDS = [
    "chest pain","difficulty breathing","shortness of breath","severe abdominal pain","severe bleeding",
    "loss of consciousness","sudden weakness","sudden numbness","slurred speech","very high fever",
    "convulsions","severe allergic reaction","anaphylaxis"
]

def detect_red_flags(text: str) -> List[str]:
    return [kw for kw in RED_FLAG_KEYWORDS if kw in text.lower()]

# ------------------ Hugging Face LLM ------------------
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    st.warning(f"Failed to load HF model, using mock responses. Error: {e}")
    generator = None

def call_llm_json(prompt: str, timeout_sec: int = 15) -> Dict[str, Any]:
    """
    Generate response from HF model. Use mock if too slow or fails.
    """
    if generator is None:
        return mock_response_for_prompt(prompt)

    try:
        start_time = time.time()
        outputs = generator(prompt, max_length=300, do_sample=True, temperature=0.2)
        elapsed = time.time() - start_time
        if elapsed > timeout_sec:
            return mock_response_for_prompt(prompt)
        text = outputs[0]["generated_text"]
    except Exception as e:
        return {"conditions": [],
                "recommendations": [f"LLM call failed: {e}"],
                "disclaimer": "Educational only",
                "red_flags": [],
                "_raw": ""}

    json_text = extract_first_json_blob(text)
    if not json_text:
        return {"conditions": [],
                "recommendations": ["Cannot parse structured output"],
                "disclaimer": "Educational only",
                "red_flags": [],
                "_raw": text}
    try:
        parsed = json.loads(json_text)
    except:
        parsed = {"conditions": [],
                  "recommendations": ["JSON parse error"],
                  "disclaimer": "Educational only",
                  "red_flags": [],
                  "_raw": text}
    parsed["_raw"] = text
    return parsed

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Healthcare Assistant", page_icon="🩺", layout="wide")
st.markdown("""
<style>
body {background-color:#f8f8f8;color:#111;font-family:'Segoe UI',sans-serif;}
h1,h2,h3{color:#111;}
.stButton>button{background-color:#333;color:white;border-radius:8px;padding:6px 12px;}
.stButton>button:hover{background-color:#555;color:white;}
.stTextArea textarea{background-color:#eee;color:#111;border-radius:5px;padding:8px;}
.progress-bar{height:16px;border-radius:8px;background:#ddd;margin-top:2px;margin-bottom:4px;}
.progress-fill-low{width:33%;background:#999;}
.progress-fill-medium{width:66%;background:#555;}
.progress-fill-high{width:100%;background:#111;}
.card{background:#fff;padding:12px;border-radius:8px;box-shadow:1px 1px 4px #ccc;margin-bottom:10px;}
</style>
""", unsafe_allow_html=True)

st.title("🩺 Healthcare Assistant")
st.caption("Educational tool — not a medical diagnosis. Use with caution.")

right_ratio = st.sidebar.slider("Adjust Right Panel Width (%)", 20,50,39)
left_ratio = 100 - right_ratio
col1, col2 = st.columns([left_ratio,right_ratio])

with col1:
    with st.form("symptom_form"):
        symptoms = st.text_area("Describe your symptoms:", height=120, placeholder="e.g., fever, sore throat, mild headache")
        col_submit1,col_submit2 = st.columns(2)
        with col_submit1: submit_btn = st.form_submit_button("Check")
        with col_submit2: sample_btn = st.form_submit_button("Sample Symptoms")

    if sample_btn:
        symptoms="Sore throat, runny nose, mild fever 100°F, no chest pain or breathing difficulty."

    if submit_btn:
        if not symptoms.strip() or len(symptoms.strip())<6:
            st.warning("Enter detailed symptom description.")
        else:
            with st.spinner("Analyzing..."):
                prompt = build_json_prompt(symptoms)
                parsed = call_llm_json(prompt)
                save_query(symptoms,parsed,raw_text=parsed.get("_raw",""))

    # Latest result
    history = fetch_history(limit=30)
    if history:
        latest = history[0]
        res = latest["response"]
        model_red_flags = res.get("red_flags",[])
        user_reds = detect_red_flags(latest["symptoms"])
        if model_red_flags or user_reds:
            st.error("⚠️ RED FLAGS detected!")
            if model_red_flags: st.write("Model flagged:",", ".join(model_red_flags))
            if user_reds: st.write("Reported:",", ".join(user_reds))
        conds=res.get("conditions",[])
        if conds:
            st.markdown("**Probable Conditions:**")
            for c in conds:
                name=c.get("name","Unknown")
                conf=c.get("confidence","low")
                notes=c.get("notes","")
                fill_class={"low":"progress-fill-low","medium":"progress-fill-medium","high":"progress-fill-high"}.get(conf,"progress-fill-low")
                st.markdown(f'<div class="card"><b>{name}</b> <i>{conf}</i><br>{notes}<div class="progress-bar"><div class="{fill_class}"></div></div></div>',unsafe_allow_html=True)
        else:
            st.info("No probable conditions. See recommendations.")

        recs=res.get("recommendations",[])
        if recs:
            st.markdown("**Recommendations:**")
            icon_map={"Rest":"💤","See clinician":"🩺","Emergency":"🚨","Consult":"⚠️"}
            for r in recs:
                icon = next((v for k,v in icon_map.items() if k.lower() in r.lower()),"ℹ️")
                st.markdown(f"{icon} {r}")

        urgent_conditions=any(flag in latest["symptoms"].lower() for flag in RED_FLAG_KEYWORDS)
        if urgent_conditions: st.warning("Consult doctor immediately for severe symptoms.")
        else: st.info("Monitor symptoms. Consult doctor if they persist or worsen.")

        st.markdown(f"**Disclaimer:** {res.get('disclaimer','Educational only.')}")
        st.caption(f"Saved at (UTC): {latest['created_at']}")

        if res.get("_raw"):
            with st.expander("Show raw model output"):
                st.code(res.get("_raw"))

with col2:
    with st.expander("📂 History & Summary", expanded=False):
        hist = fetch_history(limit=100)
        if hist:
            all_conditions=[c["name"] for h in hist for c in h["response"].get("conditions",[])]
            cond_counter = Counter(all_conditions).most_common(5)
            all_symptoms=[h["symptoms"] for h in hist]
            symptom_counter = Counter()
            for s in all_symptoms: symptom_counter.update(s.lower().split())
            top_symptoms = symptom_counter.most_common(10)

            st.markdown("### 📊 Summary Report")
            if cond_counter:
                st.markdown("**Top conditions reported:**")
                for name,count in cond_counter: st.markdown(f"- {name} ({count} times)")
            if top_symptoms: st.markdown("**Top words in symptoms:** "+", ".join([w for w,c in top_symptoms]))

            st.markdown("### 🕒 Previous Checks")
            for h in hist:
                with st.container():
                    st.markdown(f"**{h['created_at']}** — {h['symptoms'][:120]}{'...' if len(h['symptoms'])>120 else ''}")
                    col1b,col2b=st.columns([4,1])
                    with col1b:
                        if st.button(f"Show #{h['id']}",key=f"show_{h['id']}"): st.json(h["response"])
                    with col2b:
                        if st.button("Delete",key=f"del_{h['id']}"):
                            delete_query(h['id'])
                            st.success("Deleted!")
                            st.experimental_rerun()

st.markdown("---")
st.caption("Made with ❤️. Educational only. Not a substitute for medical advice.")

