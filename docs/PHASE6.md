# Phase 6: 5-Screen Interactive Streamlit Dashboard & Evidence Inspector

**Document**: `docs/PHASE6.md`  
**Phase Objective**: Build a professional, interactive 5-screen Streamlit application (`app/streamlit_app.py`) delivering an audit-ready qualitative commercial diligence workspace with dynamic matrix rendering, consensus/disagreement insights, cross-transcript conversational RAG, an evidence drawer, 1-click jump-to-line transcript inspection, and drag-and-drop document ingestion.  
**Deliverable Files**:
1. `docs/PHASE6.md` (This Specification)
2. `app/streamlit_app.py` (The 5-Screen Interactive Dashboard)

---

## 1. UI Architecture & Navigation Hierarchy

The application features a clean tabbed navigation bar across 5 core workflows:

```text
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                TRANSCRIPT INTELLIGENCE PRO                                │
│     Closed-Loop Document Intelligence for Qualitative Commercial Due Diligence           │
├───────────────┬───────────────────┬───────────────────┬───────────────────┬───────────────┤
│  1. MATRIX    │   2. SYNTHESIS    │    3. AI CHAT     │   4. INSPECTOR    │   5. UPLOAD   │
│  (6x3 Grid)   │ (Consensus/Stance)│ (Hybrid RAG Chat) │ (Provenance View) │  (Ingestion)  │
└───────────────┴───────────────────┴───────────────────┴───────────────────┴───────────────┘
```

---

## 2. Screen-by-Screen Functional Specifications

### Screen 1: Comparative Ground-Truth Matrix ($6 \times 3$)
* **Data Source**: Loaded from `storage/processed/ground_truth_matrix.json` (0 repeated LLM cost).
* **Layout**:
  - 6 rows (Questions Q1–Q6).
  - 3 columns (France: Dr. Jean Martin, Germany: Anna Keller, UK: Dr. Emily Carter).
* **Cell Component**:
  - Synthesized answer text.
  - Verification badge: `✓ EXACT_VERIFIED` (Green), `~ NORMALIZED` (Blue), `⚠ FALLBACK` (Yellow).
  - Timestamp Pills (e.g. `[02:18]`): Clicking a pill opens the **Evidence Drawer** showing the exact verbatim quote and a 1-click button: *"Jump to Line in Inspector"*.

---

### Screen 2: Cross-Expert Synthesis & Market Growth Spectrum
* **Data Source**: `storage/processed/synthesis_report.json`.
* **Layout**:
  - **Shared Consensus Themes (Left / Top Pane)**: Expandable cards displaying shared operational truths (e.g. *Single-surgeon utilization hazard*, *Bifurcated hospital adoption*).
  - **Strategic Disagreements (Right Pane)**: Comparative stakeholder cards highlighting key divergences (e.g. *Financial primacy in Germany vs Clinical balance in UK NHS*).
  - **3–5 Year Market Growth Spectrum**: Visual horizontal distribution classifying markets into **Conservative** (Germany: high-single digit), **Moderate** (France: 15–20% in top centres), and **Bullish** (UK: >15% if training expands).

---

### Screen 3: Cross-Transcript Conversational RAG
* **Data Source**: `src.core.rag_engine.RAGEngine` (Hybrid BM25 + Vector + RRF $k=60$).
* **Features**:
  - **Sidebar Filter**: Filter by Market (`All`, `France`, `Germany`, `UK`) or Role (`All`, `Clinician`, `Procurement`).
  - **Chat Interface**: Grounded answer streaming with interactive citation cards.
  - **Evidence Gate & Abstention**: If asked out-of-scope questions (e.g. *"What is the market size in Japan?"*), the system outputs honest refusal instead of fabricating facts.

---

### Screen 4: Raw Transcript Inspector
* **Data Source**: `src.models.transcript.TranscriptDocument`.
* **Features**:
  - Dropdown selector for active transcript (France, Germany, UK).
  - Line-by-line formatted transcript viewer with timestamp anchors, speaker tags, and line numbers.
  - **1-Click Jump & Highlighting**: When navigating from any citation pill elsewhere in the app, automatically sets the active transcript, scrolls to the target line, and applies visual yellow highlight animation.

---

### Screen 5: Document Upload & Dynamic Ingestion
* **Features**:
  - Drag-and-drop `.txt` transcript file uploader.
  - Validates file structure, header metadata, and timestamp integrity.
  - Calculates SHA-256 hash to ensure idempotency.
  - On upload, runs parsing, chunking, indexing, and 6-question extraction, appending the new expert column to the comparative matrix dynamically.

---

## 3. UI Aesthetics & Design Tokens

* **Theme**: Modern Dark Mode with slate/charcoal surfaces, crisp white typography, and vibrant emerald/cyan accents.
* **Badges**:
  - `EXACT_VERIFIED`: `#10b981` (Emerald green).
  - `NORMALIZED_VERIFIED`: `#0ea5e9` (Sky blue).
  - `FALLBACK_EVIDENCE`: `#f59e0b` (Amber yellow).
  - `REJECTED`: `#ef4444` (Coral red).
