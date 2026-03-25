# 🏠 Real Estate GenAI Analytics
### Powered by Claude (claude-sonnet-4-20250514)

End-to-end Generative AI pipeline that analyses real estate customer–salesman conversations using the **Anthropic Claude API** to extract:

| Output | Description |
|--------|-------------|
| 💰 Budget Range | Below 50L / 50L-1Cr / 1Cr-2Cr / Above 2Cr |
| 📍 Preferred Area | Bangalore locality |
| 🎯 Interest Level | Not Interested → Highly Interested |
| 📊 Sentiment Score | 0.0 – 1.0 float |
| 📋 Sales Stage | Closed / Not Responding / Brochure Sent / Follow-up / Site Visit |
| 🎲 Conversion Likelihood | Very Low → Very High |
| 👤 Customer Persona | First-Time Buyer / Investor / NRI / Upgrader… |
| ⏱️ Urgency | Immediate / 3 Months / 6 Months / No Urgency |
| 🔍 Key Signals | 3 specific phrases driving the assessment |
| ⚠️ Pain Points | Customer concerns identified |
| ✅ Positive Signals | Buying intent signals |
| 🎯 Recommended Action | Specific next step for the salesman |
| 🧠 AI Summary | 2–3 sentence interaction summary |

---

## Project Structure

```
realestate_genai/
├── config.py                       ← All config, labels, colors
├── requirements.txt
├── data/
│   └── raw/                        ← Excel conversation dataset
├── prompts/
│   └── analysis_system_prompt.txt  ← Claude system prompt (edit to tune)
├── src/
│   ├── genai/
│   │   ├── analyzer.py             ← Claude API integration + batch runner
│   │   └── cache.py                ← MD5-based JSON cache (avoids re-calling)
│   └── preprocessing/
│       └── data_loader.py          ← Excel loader
└── app/
    └── streamlit_app.py            ← 5-page Streamlit dashboard
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Launch dashboard
streamlit run app/streamlit_app.py
```

Or enter the API key directly in the sidebar of the Streamlit app.

---

## Streamlit Pages

| Page | Description |
|------|-------------|
| 📊 Dashboard | KPIs, charts, interest/budget/persona distributions |
| 🤖 Analyse Conversation | Single conversation → full GenAI analysis with signals, actions, summary |
| 📋 Batch Analysis | Analyse all 110 conversations in configurable batches with live progress |
| 🔬 Deep Insights | Pain points, positive signals, recommended actions, persona × budget heatmap |
| 💬 Ask the Data | Natural language Q&A over the entire analysed dataset |

---

## Key Design Decisions

### Why GenAI over traditional ML?
- **No training data needed** — Claude understands conversations out of the box
- **Richer outputs** — extracts persona, urgency, pain points, recommended actions — not just labels
- **Reasoning transparency** — every prediction comes with a natural language explanation
- **Easy to tune** — edit `prompts/analysis_system_prompt.txt` to change what's extracted

### Caching
Results are cached in `data/processed/genai_results_cache.json` by MD5 hash of the conversation. Re-running the app never calls the API for already-analysed conversations.

### Prompt Engineering
The system prompt in `prompts/analysis_system_prompt.txt` uses:
- Strict JSON output schema
- Stage consistency rules (interest → stage mapping)
- Explicit field definitions and ranges
- "Return ONLY the JSON object" to prevent prose contamination
