import os, sys, json, time
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import (INTEREST_COLORS, INTEREST_BG, INTEREST_LABELS,
                    BUDGET_LABELS, AREA_LABELS, SALES_STAGES,
                    CONVERSION_LIKELIHOOD, INTEREST_SCORE_MAP)
from src.preprocessing.data_loader import load_conversations
from src.genai.analyzer import analyse_conversation, analyse_batch
from src.genai.cache import (load_cache, set_cached, get_cached,
                              clear_cache, cache_size)
st.set_page_config(
    page_title="Real Estate GenAI Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    padding: 28px 36px; border-radius: 14px; color: white; margin-bottom: 24px;
}
.main-header h1 { color: white; margin: 0; font-size: 2rem; }
.main-header p  { color: #a8c4e0; margin: 8px 0 0; font-size: 1rem; }
.ai-badge {
    display: inline-block; background: linear-gradient(135deg,#7c3aed,#2563eb);
    color: white; padding: 3px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700; margin-left: 10px;
}
.metric-card {
    background: white; border: 1px solid #e0e7ef; border-radius: 12px;
    padding: 20px 22px; box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.metric-card h3 { font-size: 0.75rem; color: #6b7280; margin: 0 0 6px;
    text-transform: uppercase; letter-spacing: 0.05em; }
.metric-card .value { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
.signal-chip {
    display: inline-block; background: #eff6ff; color: #1d4ed8;
    border: 1px solid #bfdbfe; border-radius: 20px;
    padding: 3px 10px; font-size: 0.78rem; margin: 2px;
}
.pain-chip {
    display: inline-block; background: #fef2f2; color: #dc2626;
    border: 1px solid #fecaca; border-radius: 20px;
    padding: 3px 10px; font-size: 0.78rem; margin: 2px;
}
.positive-chip {
    display: inline-block; background: #f0fdf4; color: #16a34a;
    border: 1px solid #bbf7d0; border-radius: 20px;
    padding: 3px 10px; font-size: 0.78rem; margin: 2px;
}
.action-box {
    background: linear-gradient(135deg,#eff6ff,#f0fdf4);
    border-left: 4px solid #2563eb; border-radius: 8px;
    padding: 14px 18px; margin-top: 12px;
}
.persona-box {
    background: #faf5ff; border: 1px solid #e9d5ff; border-radius: 10px;
    padding: 14px 18px; text-align: center;
}
.summary-box {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 16px 20px; font-style: italic; color: #374151;
}
.cache-info {
    background: #f0fdf4; border: 1px solid #86efac; border-radius: 8px;
    padding: 8px 14px; font-size: 0.82rem; color: #15803d;
}
div[data-testid="stTabs"] button { font-weight: 600; }
.stProgress .st-bo { background: linear-gradient(90deg,#2563eb,#7c3aed); }
</style>
""", unsafe_allow_html=True)

def interest_badge(label: str, size: str = "normal") -> str:
    color = INTEREST_COLORS.get(label, "#6b7280")
    bg    = INTEREST_BG.get(label, "#f3f4f6")
    fs    = "0.95rem" if size == "normal" else "1.1rem"
    return (f'<span style="display:inline-block;padding:5px 14px;'
            f'border-radius:20px;font-weight:600;font-size:{fs};'
            f'color:{color};background:{bg}">{label}</span>')

def score_color(score: float) -> str:
    if score >= 0.80: return "#16a34a"
    if score >= 0.60: return "#2563eb"
    if score >= 0.35: return "#d97706"
    return "#dc2626"

def score_to_pct(score: float) -> int:
    return int(round(score * 100))

def chips(items: list, cls: str) -> str:
    if not items:
        return '<span style="color:#9ca3af;font-size:0.82rem">None detected</span>'
    return "".join(f'<span class="{cls}">{i}</span>' for i in items)


def get_api_key() -> str:
    return (st.session_state.get("api_key", "")
            or os.environ.get("GEMINI_API_KEY", ""))

def api_key_ok() -> bool:
    return bool(get_api_key().strip())


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_conversations()


with st.sidebar:
    st.markdown("### 🏠 Real Estate GenAI")
    st.markdown("---")

    # API key input
    st.markdown("**🔑 API Key**")
    key_input = st.text_input(
        "API Key", type="password",
        value=st.session_state.get("api_key", ""),
        placeholder="AIzaSy...",
        label_visibility="collapsed"
    )
    if key_input:
        st.session_state["api_key"] = key_input

    if api_key_ok():
        st.success("✅ API Key set")
    else:
        st.warning("⚠️ Enter API Key to analyse")

    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Dashboard",
        "🤖 Analyse Conversation",
        "📋 Batch Analysis",
        "🔬 Deep Insights",
        "💬 Ask the Data",
    ], label_visibility="collapsed")

    st.markdown("---")
    n_cached = cache_size()
    st.markdown(f'<div class="cache-info">🗄️ Cache: <b>{n_cached}</b> conversations analysed</div>',
                unsafe_allow_html=True)
    if st.button("🗑️ Clear Cache", use_container_width=True):
        clear_cache()
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**Model:** `Generative AI`")
    st.markdown("**Analysis per conv:** Budget · Area · Interest · Sentiment · Stage · Persona · Urgency · Action")


df = get_dataset()
cache = load_cache()



if page == "📊 Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>🏠 Real Estate GenAI Analytics</h1>
        <p>Conversation intelligence powered by Large Language Models — Budget · Area · Interest · Persona · Action</p>
    </div>""", unsafe_allow_html=True)

    # Pull cached results
    cached_results = []
    for _, row in df.iterrows():
        r = get_cached(str(row["Conversation"]), cache)
        if r:
            r["_conv_id"]       = row.get("Conv ID", "")
            r["_customer"]      = row.get("Customer Name", "")
            r["_salesman"]      = row.get("Salesman Name", "")
            r["_property"]      = row.get("Property Name", "")
            cached_results.append(r)

    if not cached_results:
        st.info("👋 No analysed conversations yet. Go to **Batch Analysis** to analyse the dataset, or **Analyse Conversation** to try a single one.")
        st.stop()

    rdf = pd.DataFrame(cached_results)

    # KPI row
    total   = len(rdf)
    hot     = (rdf["interest_level"] == "Highly Interested").sum()
    avg_s   = rdf["sentiment_score"].mean()
    top_b   = rdf["budget_range"].mode()[0] if len(rdf) else "—"

    col1, col2, col3, col4 = st.columns(4)
    kpis = [
        ("Analysed Conversations", total, "#1a1a2e"),
        ("🔥 Hot Leads", hot, "#16a34a"),
        ("Avg Sentiment Score", f"{avg_s:.2f}", "#2563eb"),
        ("Top Budget Range", top_b, "#7c3aed"),
    ]
    for col, (label, val, color) in zip([col1,col2,col3,col4], kpis):
        with col:
            st.markdown(f"""<div class="metric-card">
                <h3>{label}</h3>
                <div class="value" style="color:{color}">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Interest Level Distribution")
        ic = rdf["interest_level"].value_counts().reindex(INTEREST_LABELS[::-1]).dropna().reset_index()
        ic.columns = ["Interest", "Count"]
        fig = px.bar(ic, x="Count", y="Interest", orientation="h",
                     color="Interest",
                     color_discrete_map=INTEREST_COLORS,
                     template="plotly_white")
        fig.update_layout(showlegend=False, height=300,
                          margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Budget Range Distribution")
        bc = rdf["budget_range"].value_counts().reset_index()
        bc.columns = ["Budget", "Count"]
        fig2 = px.pie(bc, values="Count", names="Budget", hole=0.45,
                      color_discrete_sequence=["#1a1a2e","#2563eb","#60a5fa","#bfdbfe"],
                      template="plotly_white")
        fig2.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Customer Persona Mix")
        pc = rdf["customer_persona"].value_counts().head(8).reset_index()
        pc.columns = ["Persona", "Count"]
        fig3 = px.bar(pc, x="Count", y="Persona", orientation="h",
                      color_discrete_sequence=["#7c3aed"],
                      template="plotly_white")
        fig3.update_layout(height=320, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown("#### Urgency Distribution")
        uc = rdf["urgency"].value_counts().reset_index()
        uc.columns = ["Urgency", "Count"]
        colors_u = {"Immediate":"#dc2626","Within 3 Months":"#ea580c",
                    "Within 6 Months":"#d97706","No Urgency":"#6b7280","Unknown":"#d1d5db"}
        fig4 = px.pie(uc, values="Count", names="Urgency", hole=0.4,
                      color="Urgency", color_discrete_map=colors_u,
                      template="plotly_white")
        fig4.update_layout(height=320, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig4, use_container_width=True)

    # Sentiment scatter
    st.markdown("#### Sentiment Score by Interest Level")
    fig5 = px.strip(rdf, x="interest_level", y="sentiment_score",
                    color="interest_level",
                    color_discrete_map=INTEREST_COLORS,
                    template="plotly_white",
                    category_orders={"interest_level": INTEREST_LABELS},
                    labels={"interest_level":"Interest Level",
                            "sentiment_score":"Sentiment Score"})
    fig5.update_layout(showlegend=False, height=320,
                       margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig5, use_container_width=True)



elif page == "🤖 Analyse Conversation":
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Conversation Analyser</h1>
        <p>Paste any customer–salesman conversation for instant GenAI-powered intelligence</p>
    </div>""", unsafe_allow_html=True)

    if not api_key_ok():
        st.error("⚠️ Please enter your API key in the sidebar.")
        st.stop()


    conv_ids = df["Conv ID"].tolist()
    selected = st.selectbox("📂 Load a sample conversation",
                            ["— Paste your own below —"] + conv_ids)
    default_text = ""
    if selected != "— Paste your own below —":
        idx = conv_ids.index(selected)
        default_text = df["Conversation"].iloc[idx]

    conv_input = st.text_area("💬 Conversation", value=default_text, height=280,
                               placeholder="[Salesman]: ...\n[Customer]: ...")

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        analyse_btn = st.button("🚀 Analyse with AI", type="primary",
                                use_container_width=True)

    if analyse_btn:
        if not conv_input.strip():
            st.warning("Please enter a conversation.")
        else:
            # Check cache first
            cached = get_cached(conv_input, cache)
            if cached:
                result = cached
                st.info("⚡ Loaded from cache (no API call made)")
            else:
                with st.spinner("🧠 AI is analysing the conversation…"):
                    result = analyse_conversation(conv_input, api_key=get_api_key())
                    cache  = set_cached(conv_input, result, cache)

            if result.get("_error"):
                st.error(f"Analysis error: {result['_error']}")
            else:
                st.markdown("---")
                st.markdown("### 📊 GenAI Analysis Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**💰 Budget Range**")
                    st.markdown(f"""<div style='font-size:1.8rem;font-weight:700;
                        color:#1a1a2e;margin:8px 0'>{result['budget_range']}</div>""",
                        unsafe_allow_html=True)
                    st.caption(f"📝 {result.get('budget_reasoning','')}")

                with col2:
                    st.markdown("**📍 Preferred Area**")
                    st.markdown(f"""<div style='font-size:1.8rem;font-weight:700;
                        color:#1a1a2e;margin:8px 0'>{result['preferred_area']}</div>""",
                        unsafe_allow_html=True)
                    st.caption(f"📝 {result.get('area_reasoning','')}")

                with col3:
                    st.markdown("**🎯 Interest Level**")
                    st.markdown(interest_badge(result["interest_level"], "large"),
                                unsafe_allow_html=True)
                    score = result["sentiment_score"]
                    color = score_color(score)
                    st.markdown(f"""<div style='margin-top:10px'>
                        <div style='font-size:0.8rem;color:#6b7280'>Sentiment Score</div>
                        <div style='font-size:2rem;font-weight:700;color:{color}'>{score:.2f}</div>
                        <div style='background:#e5e7eb;border-radius:5px;height:8px;margin-top:4px'>
                          <div style='width:{score_to_pct(score)}%;background:{color};
                          height:8px;border-radius:5px'></div>
                        </div></div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                col4, col5, col6, col7 = st.columns(4)
                with col4:
                    st.metric("📋 Sales Stage", result["sales_stage"])
                with col5:
                    st.metric("🎲 Conversion Likelihood", result["conversion_likelihood"])
                with col6:
                    st.metric("👤 Customer Persona", result.get("customer_persona","—"))
                with col7:
                    st.metric("⏱️ Urgency", result.get("urgency","—"))

                st.markdown("<br>", unsafe_allow_html=True)

                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.markdown("**🔍 Key Signals**")
                    st.markdown(chips(result.get("key_signals",[]), "signal-chip"),
                                unsafe_allow_html=True)
                with sc2:
                    st.markdown("**✅ Positive Signals**")
                    st.markdown(chips(result.get("positive_signals",[]), "positive-chip"),
                                unsafe_allow_html=True)
                with sc3:
                    st.markdown("**⚠️ Pain Points**")
                    st.markdown(chips(result.get("pain_points",[]), "pain-chip"),
                                unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Recommended Action ─────────────────────────────────────
                st.markdown(f"""<div class="action-box">
                    <b>🎯 Recommended Action</b><br>
                    {result.get('recommended_action','—')}
                </div>""", unsafe_allow_html=True)

                # ── AI Summary ─────────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""<div class="summary-box">
                    <b>🧠 AI Summary</b><br><br>
                    {result.get('summary','—')}
                </div>""", unsafe_allow_html=True)

                # ── Raw JSON expander ──────────────────────────────────────
                with st.expander("🔧 Raw JSON Response"):
                    st.json({k:v for k,v in result.items()
                             if not k.startswith("_")})



elif page == "📋 Batch Analysis":
    st.markdown("""
    <div class="main-header">
        <h1>📋 Batch AI Analysis</h1>
        <p>Analyse all 110 conversations with GenAI — results cached automatically</p>
    </div>""", unsafe_allow_html=True)

    if not api_key_ok():
        st.error("⚠️ Please enter your API key in the sidebar.")
        st.stop()

    # Status
    n_total  = len(df)
    n_done   = sum(1 for _, row in df.iterrows()
                   if get_cached(str(row["Conversation"]), cache))
    n_remain = n_total - n_done

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Total Conversations", n_total)
    with colB:
        st.metric("✅ Already Analysed", n_done)
    with colC:
        st.metric("⏳ Remaining", n_remain)

    st.progress(n_done / n_total if n_total else 0)

    # Batch size selector
    batch_n = st.slider("Conversations to analyse now", 1,
                        min(n_remain, 50) if n_remain > 0 else 1,
                        min(10, n_remain) if n_remain > 0 else 1,
                        disabled=(n_remain == 0))

    if n_remain == 0:
        st.success("✅ All conversations have been analysed! View results in Dashboard or below.")
    else:
        if st.button(f"🚀 Analyse next {batch_n} conversations", type="primary",
                     use_container_width=True):
            pending = [(i, row) for i, row in df.iterrows()
                       if not get_cached(str(row["Conversation"]), cache)][:batch_n]

            progress_bar = st.progress(0)
            status_text  = st.empty()
            results_live = st.empty()

            for done_idx, (i, row) in enumerate(pending):
                conv = str(row["Conversation"])
                cid  = row.get("Conv ID", f"#{i}")
                status_text.markdown(f"🔄 Analysing **{cid}** ({done_idx+1}/{len(pending)})…")

                result = analyse_conversation(conv, api_key=get_api_key())
                cache  = set_cached(conv, result, cache)

                progress_bar.progress((done_idx + 1) / len(pending))

                # Live mini-result
                il    = result["interest_level"]
                score = result["sentiment_score"]
                results_live.markdown(
                    f"✅ `{cid}` → **{il}** | Sentiment: {score:.2f} | "
                    f"Budget: {result['budget_range']} | Area: {result['preferred_area']}"
                )
                time.sleep(0.2)

            status_text.success(f"✅ Analysed {len(pending)} conversations!")
            st.rerun()

    # Results table
    st.markdown("---")
    st.markdown("### 📊 Analysis Results")

    rows = []
    for _, row in df.iterrows():
        r = get_cached(str(row["Conversation"]), cache)
        if r:
            rows.append({
                "Conv ID":       row.get("Conv ID",""),
                "Customer":      row.get("Customer Name",""),
                "Salesman":      row.get("Salesman Name",""),
                "Property":      row.get("Property Name",""),
                "Budget Range":  r["budget_range"],
                "Pref. Area":    r["preferred_area"],
                "Interest":      r["interest_level"],
                "Sentiment":     r["sentiment_score"],
                "Sales Stage":   r["sales_stage"],
                "Likelihood":    r["conversion_likelihood"],
                "Persona":       r.get("customer_persona",""),
                "Urgency":       r.get("urgency",""),
            })

    if rows:
        result_df = pd.DataFrame(rows)

        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            f_int = st.multiselect("Filter Interest",
                options=INTEREST_LABELS, default=[])
        with fc2:
            f_bud = st.multiselect("Filter Budget",
                options=BUDGET_LABELS, default=[])
        with fc3:
            f_per = st.multiselect("Filter Persona",
                options=result_df["Persona"].unique().tolist(), default=[])

        filtered = result_df.copy()
        if f_int: filtered = filtered[filtered["Interest"].isin(f_int)]
        if f_bud: filtered = filtered[filtered["Budget Range"].isin(f_bud)]
        if f_per: filtered = filtered[filtered["Persona"].isin(f_per)]

        st.markdown(f"**{len(filtered)} records**")
        st.dataframe(filtered.reset_index(drop=True),
                     use_container_width=True, height=480)

        csv = filtered.to_csv(index=False)
        st.download_button("⬇️ Download CSV", csv,
                           "genai_analysis_results.csv", "text/csv",
                           use_container_width=True)
    else:
        st.info("No results yet. Run batch analysis above.")

elif page == "🔬 Deep Insights":
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Deep Insights</h1>
        <p>AI-extracted patterns — pain points, positive signals, recommended actions</p>
    </div>""", unsafe_allow_html=True)

    rows = []
    for _, row in df.iterrows():
        r = get_cached(str(row["Conversation"]), cache)
        if r:
            rows.append(r)

    if not rows:
        st.info("No analysed conversations yet. Run Batch Analysis first.")
        st.stop()

    rdf = pd.DataFrame(rows)

    tab1, tab2, tab3, tab4 = st.tabs(["Pain Points", "Positive Signals",
                                       "Recommended Actions", "Persona × Budget"])

    with tab1:
        st.markdown("#### Most Common Customer Pain Points (AI Extracted)")
        all_pains = []
        for r in rows:
            all_pains.extend(r.get("pain_points", []))
        if all_pains:
            pain_counts = pd.Series(all_pains).value_counts().head(15).reset_index()
            pain_counts.columns = ["Pain Point", "Count"]
            fig = px.bar(pain_counts, x="Count", y="Pain Point", orientation="h",
                         color_discrete_sequence=["#dc2626"], template="plotly_white")
            fig.update_layout(height=450, margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No pain points extracted yet.")

    with tab2:
        st.markdown("#### Most Common Positive Signals (AI Extracted)")
        all_pos = []
        for r in rows:
            all_pos.extend(r.get("positive_signals", []))
        if all_pos:
            pos_counts = pd.Series(all_pos).value_counts().head(15).reset_index()
            pos_counts.columns = ["Positive Signal", "Count"]
            fig2 = px.bar(pos_counts, x="Count", y="Positive Signal", orientation="h",
                          color_discrete_sequence=["#16a34a"], template="plotly_white")
            fig2.update_layout(height=450, margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No positive signals extracted yet.")

    with tab3:
        st.markdown("#### AI Recommended Actions by Interest Level")
        action_rows = []
        for r in rows:
            action_rows.append({
                "Interest":  r.get("interest_level",""),
                "Action":    r.get("recommended_action",""),
                "Persona":   r.get("customer_persona",""),
                "Urgency":   r.get("urgency",""),
            })
        if action_rows:
            adf = pd.DataFrame(action_rows)
            for il in INTEREST_LABELS[::-1]:
                sub = adf[adf["Interest"] == il]
                if sub.empty:
                    continue
                color = INTEREST_COLORS.get(il, "#6b7280")
                bg    = INTEREST_BG.get(il, "#f3f4f6")
                st.markdown(
                    f'<div style="background:{bg};border-left:4px solid {color};'
                    f'border-radius:8px;padding:10px 16px;margin-bottom:8px">'
                    f'<b style="color:{color}">{il}</b> — {len(sub)} customers<br>'
                    f'<small>Sample action: {sub["Action"].iloc[0]}</small>'
                    f'</div>', unsafe_allow_html=True)
        else:
            st.info("No actions extracted yet.")

    with tab4:
        st.markdown("#### Persona Distribution by Budget Range")
        pr_rows = []
        for r in rows:
            pr_rows.append({"Persona": r.get("customer_persona","Unknown"),
                            "Budget": r.get("budget_range","Unknown")})
        if pr_rows:
            prdf = pd.DataFrame(pr_rows)
            ct   = pd.crosstab(prdf["Persona"], prdf["Budget"])
            fig3 = px.imshow(ct, color_continuous_scale="Blues", text_auto=True,
                             template="plotly_white",
                             labels={"x":"Budget Range","y":"Persona","color":"Count"})
            fig3.update_layout(height=420, margin=dict(t=20,b=10))
            st.plotly_chart(fig3, use_container_width=True)


elif page == "💬 Ask the Data":
    st.markdown("""
    <div class="main-header">
        <h1>💬 Ask the Data</h1>
        <p>Ask natural language questions about your conversation dataset — AI answers using the cached analysis</p>
    </div>""", unsafe_allow_html=True)

    if not api_key_ok():
        st.error("⚠️ Please enter your API key in the sidebar.")
        st.stop()

    rows = []
    for _, row in df.iterrows():
        r = get_cached(str(row["Conversation"]), cache)
        if r:
            rows.append({
                "conv_id":       row.get("Conv ID",""),
                "customer":      row.get("Customer Name",""),
                "salesman":      row.get("Salesman Name",""),
                "property":      row.get("Property Name",""),
                "budget_range":  r.get("budget_range",""),
                "preferred_area":r.get("preferred_area",""),
                "interest_level":r.get("interest_level",""),
                "sentiment_score":r.get("sentiment_score",""),
                "sales_stage":   r.get("sales_stage",""),
                "conversion_likelihood": r.get("conversion_likelihood",""),
                "customer_persona": r.get("customer_persona",""),
                "urgency":       r.get("urgency",""),
                "recommended_action": r.get("recommended_action",""),
                "pain_points":   r.get("pain_points",[]),
                "positive_signals": r.get("positive_signals",[]),
                "summary":       r.get("summary",""),
            })

    if not rows:
        st.info("No analysed conversations yet. Run Batch Analysis first.")
        st.stop()

    # Suggested questions
    st.markdown("**💡 Try asking:**")
    suggestions = [
        "Which customers are most likely to convert?",
        "What are the top pain points across all customers?",
        "Which salesman has the most hot leads?",
        "What budget range is most common?",
        "Which areas are most in demand?",
        "Who should I follow up with today?",
    ]
    cols = st.columns(3)
    for i, sug in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(sug, use_container_width=True):
                st.session_state["ask_question"] = sug

    question = st.text_input(
        "Ask a question about your data",
        value=st.session_state.get("ask_question", ""),
        placeholder="e.g. Which customers are ready to book a site visit?"
    )

    if question and st.button("🧠 Get AI Answer", type="primary"):
        # Build a compact data summary for the prompt
        summary_lines = []
        for r in rows:
            summary_lines.append(
                f"- {r['customer']} | {r['budget_range']} | {r['preferred_area']} | "
                f"{r['interest_level']} | Score:{r['sentiment_score']} | "
                f"Persona:{r['customer_persona']} | Urgency:{r['urgency']} | "
                f"Stage:{r['sales_stage']} | Action:{r['recommended_action']}"
            )
        data_str = "\n".join(summary_lines[:80])  # cap at 80 rows

        prompt = f"""You are a real estate sales analytics expert.
Below is a summary of {len(rows)} customer conversations that have been AI-analysed.
Each line: Customer | Budget | Area | Interest Level | Sentiment Score | Persona | Urgency | Sales Stage | Recommended Action

DATA:
{data_str}

QUESTION: {question}

Provide a clear, concise, actionable answer based on the data above.
Use bullet points where appropriate. Be specific — name customers, quote numbers."""

        with st.spinner("🧠 AI is analysing your dataset…"):
            from src.genai.analyzer import GEMINI_API_URL
            import requests as req
            from config import GEMINI_MODEL
            headers = {
                "Content-Type": "application/json",
            }
            url = f"{GEMINI_API_URL}{GEMINI_MODEL}:generateContent?key={get_api_key()}"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 4096,
                }
            }
            try:
                resp = req.post(url, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                data_r = resp.json()
                answer = data_r.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                
                st.markdown("---")
                st.markdown("### 🤖 AI Answer")
                st.markdown(f"""<div style='background:#f8fafc;border:1px solid #e2e8f0;
                    border-radius:12px;padding:20px 24px;line-height:1.8'>
                    {answer.replace(chr(10),'<br>')}
                </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
