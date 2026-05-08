# Grant Materials Update (M3/M4 Empirical Focus) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update grant materials to reflect M3/M4 empirical progress.

**Architecture:** Targetted updates to Markdown documentation files.

**Tech Stack:** Markdown.

---

### Task 1: Update Executive Abstract

**Files:**
- Modify: `docs/grant_materials/Executive_Abstract.md`

- [ ] **Step 1: Update "Key Innovations" section**
Update the Thalamic Gating section to include room-colored prediction and speaker-embedding.

- [ ] **Step 2: Update "Current Progress & Evidence" section**
Add Agentive Turn-Taking (M3) and Self-Modeling Impact (M4) subsections.

- [ ] **Step 3: Verify changes**
Ensure the text matches the data from `docs/m3_start.md` and `docs/m4_forward_model_state.md`.

- [ ] **Step 4: Commit**
```bash
git add docs/grant_materials/Executive_Abstract.md
git commit -m "docs(grant): update executive abstract with M3/M4 results"
```

### Task 2: Update Research Methodology

**Files:**
- Modify: `docs/grant_materials/Research_Methodology.md`

- [ ] **Step 1: Update "Developmental Phasing"**
Update Stage III and IV descriptions.

- [ ] **Step 2: Update "Experimental Paradigms"**
Add the Embodied TTS-to-Mic Loop.

- [ ] **Step 3: Update "Core Validation Metrics"**
Add Interruption Recovery Rate and M4 Self-Attribution Gain.

- [ ] **Step 4: Verify changes**
Ensure consistency with the implementation notes in `docs/m3_start.md`.

- [ ] **Step 5: Commit**
```bash
git add docs/grant_materials/Research_Methodology.md
git commit -m "docs(grant): update research methodology with M3/M4 metrics"
```

### Task 3: Update Visual Data Summary

**Files:**
- Modify: `docs/grant_materials/Visual_Data_Summary.md`

- [ ] **Step 1: Update Figure 4 (Agentive Contingency)**
Point to `scripts/m3_turn_taking_demo.py` and describe the timing diagram.

- [ ] **Step 2: Add Figure 5 (M4 Self-Modeling Impact)**
Add the description for the M4 baseline vs M4-enabled comparison.

- [ ] **Step 3: Verify changes**
Ensure the figure descriptions accurately represent the results in `docs/m3_start.md`.

- [ ] **Step 4: Commit**
```bash
git add docs/grant_materials/Visual_Data_Summary.md
git commit -m "docs(grant): update visual data summary with M3/M4 figures"
```
