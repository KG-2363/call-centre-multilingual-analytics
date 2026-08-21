"""CCMA dashboard.

Run:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `src/ccma` and the sibling dashboard modules importable when Streamlit is
# invoked from the repo root without an editable install.
_ROOT = Path(__file__).resolve().parent.parent
for path in (_ROOT / "src", _ROOT / "dashboard"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import data as ds  # noqa: E402
import theme as th  # noqa: E402
import views  # noqa: E402

st.set_page_config(
    page_title="Call Centre Multilingual Analytics",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

th.register_template()
st.markdown(th.CSS, unsafe_allow_html=True)

VIEWS = {
    "Executive KPIs": views.executive,
    "Transcript explorer": views.transcript_explorer,
    "Language & code-switching": views.language_view,
    "Sentiment": views.sentiment_view,
    "Agent scorecards": views.agent_scorecards,
    "Compliance monitoring": views.compliance_view,
    "Escalation alerts": views.escalation_view,
}

with st.sidebar:
    st.markdown("### Call Centre Analytics")
    st.caption("English · हिन्दी · Hinglish · romanised Hindi")
    selection = st.radio("View", list(VIEWS), label_visibility="collapsed")
    st.divider()

    if ds.warehouse_ready():
        calls = ds.call_ids()
        st.metric("Calls in warehouse", f"{len(calls):,}")
        st.caption(f"Latest: {calls['recorded_at'].max()}" if not calls.empty else "")
    st.divider()
    st.caption(
        "Every KPI is computed over quality-eligible calls only. Where a metric "
        "depends on ASR or LLM output, its caveat is shown next to the number."
    )
    if st.button("Clear cache"):
        st.cache_data.clear()
        st.rerun()

st.title("Multilingual Call-Centre Analytics")

if not ds.warehouse_ready():
    st.error("Warehouse not built.")
    st.code("pip install -r requirements.txt\npip install -e .\nccma run-all", language="bash")
    st.stop()

VIEWS[selection]()
