# The Best Way to Automate a Continuous, Worldwide, Human-in-the-Loop Job Hunt for Yogesh Gandhi

*Decision-grade research report. Prepared 2026-06-25. Candidate: PhD Aerospace/computational mechanics (topology optimization, nonlinear FEM, composites), ~7 yrs research, first-author CMAME, currently Senior Researcher at Braid Technologies (Tokyo). Needs employer visa sponsorship; will relocate anywhere; targets industry (aerospace/automotive/F1/CAE-software vendors), national labs/R&D, and academia; wants quality over cost; wants human review before every submit (no autonomous auto-apply).*

---

## 1. Executive summary

- **Build, don't buy a turnkey product.** No off-the-shelf "AI job copilot" fits this profile: they are US/tech-skewed, lack a real worldwide visa-sponsorship filter, and their matching is tuned for mainstream roles — weak exactly where a CMAME-publishing computational-mechanics specialist needs precision. Yogesh is a Python/JAX engineer building a simulation engine; a custom pipeline is well within reach and wins on the two stages that matter most (niche matching and tailoring) [job-fit, end-to-end, cv-cover-letter dossiers].
- **The backbone is free and zero-legal-risk: curated ATS public JSON feeds.** Most of his named targets (Ansys/Dassault/Altair/Hexagon/COMSOL, F1 teams, labs) run Greenhouse / Lever / Ashby / SmartRecruiters / Workday boards with unauthenticated public endpoints. Curating a `{employer → ATS + token}` map is the highest-ROI component [1][2][3][9].
- **Whole-web breadth comes from SerpApi Google Jobs** (free 250 searches/mo; $75/mo = 5,000), which uniquely covers Japan/Asia where Adzuna is blind, plus **Adzuna** (free, 12 countries, salary normalization) and a **licensed aggregator (Active Jobs DB / Fantastic Jobs)** for LinkedIn-sourced coverage *without* scraping LinkedIn [4][5][6][7].
- **Visa sponsorship must be a separate gating layer, never blended into the match score.** Use authoritative government sponsor registers (UK licensed-sponsor CSV, Netherlands IND recognised sponsors, Canada LMIA lists, US H-1B/LCA) + sponsorship-specific boards (Arbeitnow `?visa_sponsorship=true`, TokyoDev/JapanDev for his current base) + an EU Blue Card rule engine (as a PhD engineer he clears every threshold) [22][23][24][25][26][29][30].
- **Matching = two-stage retrieve-and-rerank with an LLM-judge** over a hand-tuned FEM/CAE rubric that emits explainable sub-scores and a keyword-gap report (which feeds tailoring). Cost is trivial (cents/day); the IP is the rubric [job-fit dossier].
- **Tailoring = a DIY Claude (Opus 4.8) pipeline that selects/reorders from a verified YAML source-of-truth** (never fabricates), validated against an ATS scorer (Jobscan or open-source Resume-Matcher), targeting a 78-82% match (never push past ~85% — that reads as keyword stuffing) [10][11][12][14].
- **Human-in-the-loop is both his rule and the safe architecture.** Keep the final Submit a manual human click; never script submission; never automate LinkedIn under any account. This satisfies his constraint AND sidesteps the contract/ban risk that has destroyed scrapers (hiQ → $500k judgment + permanent injunction; Proxycurl shut down Jul 2025) [15][16][17].
- **Recommended runtime: LangGraph `interrupt()` + checkpointer in Python** (matches his stack), with an Airtable/Notion kanban as system-of-record and a once-daily 9am digest for review. Estimated all-in running cost ~**$120-260/month** at "best, not cheapest" settings (details in §9) [human-in-loop dossier].

---

## 2. Recommended architecture — concrete end-to-end pipeline

### Design principles
1. **Precision over volume.** For a niche senior profile, 30 well-matched sponsor-eligible roles/month beat 3,000 generic ones.
2. **Sponsorship is a gate, not a score.** A perfect-fit role that won't sponsor must be flagged/deprioritized, never ranked #1.
3. **Lazy generation.** Cheap match-score + "why-fit" bullets in the digest; the expensive ATS-tailored CV + cover letter is generated *only after he approves the match*.
4. **Human disposes.** Agent proposes; he approves/edits/rejects; he clicks Submit on the employer's own ATS.

### Flow diagram

```
                         ┌─────────────────────────────────────────────────────┐
                         │            DAILY SCHEDULED RUN (cron)                 │
                         └─────────────────────────────────────────────────────┘
                                              │
   ┌──────────────────────────────────────────────────────────────────────────────────────┐
   │ STAGE 1 — SEARCH (whole web, worldwide)                                                 │
   │                                                                                         │
   │  BACKBONE (free, high precision)        BREADTH (whole web)        SPONSORSHIP-SPECIFIC │
   │  • Greenhouse/Lever/Ashby/Smart-        • SerpApi Google Jobs      • Arbeitnow API      │
   │    Recruiters/Workday CXS feeds for       (incl. Japan/Asia)         (?visa_sponsorship)│
   │    curated target employers             • Adzuna (12 countries)    • TokyoDev/JapanDev  │
   │  • jobs.ac.uk RSS (UK academia)         • Active Jobs DB           • EURAXESS scrape     │
   │  • USAJOBS API (US fed/labs)              (licensed LinkedIn)      • Relocate.me / DLR   │
   └──────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                          ┌───────────────────────────────────────┐
                          │ STAGE 2 — NORMALIZE + DEDUP            │
                          │ canonical schema; fuzzy title+company  │
                          │ dedup (rapidfuzz); drop expired        │
                          └───────────────────────────────────────┘
                                              │
                 ┌──────────────────────────────────────────────────────────┐
                 │ STAGE 3 — SPONSORSHIP / LOCATION GATE  (separate signal)   │
                 │ • fuzzy-match employer vs gov sponsor registers           │
                 │   (UK CSV / NL IND / CA LMIA / US H-1B+LCA)               │
                 │ • parse JD for sponsorship language                       │
                 │ • EU Blue Card rule engine (PhD ⇒ eligible)               │
                 │ → emit sponsorship_signal {yes|likely|unknown|no}         │
                 │ → hard-exclude US-citizenship-only (national labs)        │
                 └──────────────────────────────────────────────────────────┘
                                              │
        ┌────────────────────────────────────────────────────────────────────────┐
        │ STAGE 4 — MATCH / RANK  (two-stage retrieve-and-rerank)                   │
        │ 4a. bi-encoder embeddings (BGE/E5/Voyage) over WHOLE corpus → top-N       │
        │ 4b. LLM-as-judge (Claude) on top-N with FEM/CAE rubric:                   │
        │     method_fit · software_fit · domain_fit · seniority · research/product │
        │     → overall 0-100 + sub-scores + rationale + top-3 missing keywords     │
        └────────────────────────────────────────────────────────────────────────┘
                                              │
                 ┌──────────────────────────────────────────────────────┐
                 │ STAGE 5 — SURFACE (human-in-the-loop INTERRUPT #1)    │
                 │ LangGraph interrupt() → write top-N to Airtable kanban │
                 │ + 9am digest (email/Telegram). Evidence pack per item: │
                 │ match breakdown · WHY-fit bullets · SPONSORSHIP FLAG · │
                 │ salary · location · freshness · apply link             │
                 │ Human action: APPROVE / EDIT / REJECT                  │
                 └──────────────────────────────────────────────────────┘
                                              │ (on APPROVE only)
                 ┌──────────────────────────────────────────────────────┐
                 │ STAGE 6 — TAILOR (lazy generation)                    │
                 │ Claude Opus 4.8: select/reorder from verified         │
                 │ resume.yaml (NO fabrication, Pydantic-validated) →    │
                 │ tailored CV + cover letter + ATS check (Jobscan /     │
                 │ Resume-Matcher), target 78-82% match                  │
                 └──────────────────────────────────────────────────────┘
                                              │
                 ┌──────────────────────────────────────────────────────┐
                 │ STAGE 7 — REVIEW (human-in-the-loop INTERRUPT #2)     │
                 │ Show CV+cover letter + diff vs base + ATS score.      │
                 │ Human: approve / edit / regenerate                    │
                 └──────────────────────────────────────────────────────┘
                                              │
                 ┌──────────────────────────────────────────────────────┐
                 │ STAGE 8 — ASSISTED APPLY (HUMAN CLICKS SUBMIT)        │
                 │ Simplify Copilot autofill on employer ATS;            │
                 │ HE answers visa/work-auth + clicks Submit himself.    │
                 └──────────────────────────────────────────────────────┘
                                              │
                 ┌──────────────────────────────────────────────────────┐
                 │ STAGE 9 — TRACK                                       │
                 │ Airtable/Huntr kanban: Applied→Interview→Offer;       │
                 │ reject reasons feed back into ranking rubric          │
                 └──────────────────────────────────────────────────────┘
```

### Tool choice per stage and why

| Stage | Recommended tool/service | Why for this candidate |
|---|---|---|
| **1. Search — backbone** | ATS public feeds: Greenhouse `boards-api.greenhouse.io/v1/boards/{token}/jobs`, Lever `api.lever.co/v0/postings/{client}`, Ashby (`?includeCompensation=true`), SmartRecruiters `/v1/companies/{id}/postings`, Workday CXS | Free, no auth, highest signal-to-noise; directly hits his named CAE/aerospace/lab targets. Ashby/Lever expose salary [1][2][3][9]. |
| **1. Search — UK academia** | jobs.ac.uk native RSS (`/feeds/subject-areas/engineering-and-technology`) | Cleanest legal source for UK academic FEM/aerospace roles; strong sponsorship signal (Skilled Worker/Global Talent) [31]. |
| **1. Search — US fed/labs** | USAJOBS REST API (free key) | Covers NASA + national labs. Low weight: most need US citizenship — keep for completeness, flag as hard exclusion [8][32]. |
| **1. Search — breadth** | SerpApi Google Jobs API | One call, worldwide, **covers Japan/Asia** where Adzuna is blind; predictable pricing [5]. |
| **1. Search — breadth supplement** | Adzuna API (free) | 12 countries + salary normalization; free [6]. |
| **1. Search — LinkedIn w/o risk** | Active Jobs DB / Fantastic Jobs (RapidAPI) | Aggregated LinkedIn/Wellfound/YC + 54 ATS, hourly refresh — consume LinkedIn data without scraping it yourself [7]. |
| **2. Dedup** | rapidfuzz / custom | Fuzzy company+title dedup across overlapping sources. |
| **3. Sponsorship gate** | UK gov CSV [22], NL IND register [23], Canada LMIA [24], US DOL OFLC + USCIS H-1B Hub [25][41], Arbeitnow `?visa_sponsorship=true` [29], EU Blue Card rule engine [27] | Turns sponsorship from an ignored variable into a first-class, human-reviewable flag — *the* deal-breaker for him. |
| **4. Match/rank** | bi-encoder (BGE/E5/Voyage) + Claude LLM-judge with FEM/CAE rubric; reuse srbhr/Resume-Matcher as per-job scorer | Only approach that correctly weights rare tokens (LS-DYNA, geometry-projection topology opt, IGA/NURBS, CMAME) and emits gap data [33][34]. |
| **5/7. HITL gate** | LangGraph `interrupt()` + checkpointer | Python-native (his stack); durable across day-long waits; gates only the irreversible submit [35]. |
| **5. Surfacing** | Airtable kanban (native automations) + 9am digest via Telegram/email | Durable system-of-record; once-daily batch avoids notification fatigue [37]. |
| **6. Tailor** | Claude Opus 4.8 pipeline over Pydantic-validated `resume.yaml`; fork claude-code-job-tailor / resume-tailoring-skill | Per-job reasoning beats generic SaaS prompts; schema validation prevents fabrication of skills/metrics [10][11][12]. |
| **6. ATS check** | Jobscan (free 5/mo or $49.95/mo) OR Resume-Matcher (free, local) | Validate 78-82% target; Resume-Matcher fits his build-it ethos and keeps data local [13][14]. |
| **8. Assisted apply** | Simplify Copilot (free autofill) | He clicks Submit; visa/work-auth answers always correct; TOS-safe (user-click only) [18][39]. |
| **9. Track** | Airtable (or Huntr if he wants docs layer off-the-shelf) | Kanban tracker; reject reasons retrain ranking [38]. |

---

## 3. Two alternative builds

### (a) Low-effort no-code: n8n/Make + off-the-shelf tools
- **Orchestration:** n8n (`sendAndWait` approval node; prebuilt Slack #5049 / Telegram #6026 templates; n8n+Claude+Google Sheets job-match template) [40][36].
- **Search:** Adzuna + Arbeitnow + SerpApi nodes → Google Sheets.
- **Match:** Claude node scores each JD vs CV; writes fit score to sheet.
- **Surface/approve:** n8n `sendAndWait` → Telegram approve/reject → on approve, Claude drafts CV+cover letter.
- **Apply:** Simplify Copilot manual autofill.
- **Caveat:** verify n8n `sendAndWait` reliability on his version (documented Slack loop/resend and token bugs historically) [40].

### (b) Custom-code: Python + Claude API + LangGraph (+ Playwright only for scraping, not submit)
- **Orchestration:** LangGraph `interrupt()` + checkpointer (SQLite/Postgres) [35].
- **Search:** async httpx clients for ATS feeds + SerpApi/Adzuna/Arbeitnow; Playwright **only** for the irreplaceable scrape targets (EURAXESS `/jobs/search`, DLR) — never against LinkedIn.
- **Match:** sentence-transformers bi-encoder + Claude LLM-judge rubric; reuse Resume-Matcher.
- **Tailor:** Claude Opus 4.8 over `resume.yaml` (fork claude-code-job-tailor).
- **Apply:** Simplify Copilot (manual) — do **not** automate the submit click.

### Trade-offs

| Dimension | (a) No-code n8n/Make | (b) Custom Python/LangGraph | Recommended hybrid |
|---|---|---|---|
| Build effort | Low (days) | High (weeks) | Medium |
| Niche-match quality | Medium (generic Claude prompts) | High (tuned FEM rubric) | High |
| Sponsorship gating | Hard to encode cleanly | Full control | Full control |
| Maintenance | Vendor node drift; sendAndWait bugs | He owns it (he can maintain) | He owns the core |
| Fit to his skills | Under-uses his Python depth | Perfect fit | Perfect fit |
| Verdict | Good prototype / fallback | **Best engine** | **Custom core + Airtable/Simplify off-the-shelf glue** |

The candidate's profile (Python/JAX engineer at a simulation-automation startup) argues decisively for **(b) as the core**, with off-the-shelf components (Airtable kanban, Simplify autofill, Jobscan) bolted on rather than rebuilt.

---

## 4. Worldwide job-source strategy + visa-sponsorship filtering + niche CAE/research sources

### Worldwide sources (tiered)
- **Tier 1 — Backbone (free, precise):** curated ATS feeds for named employers (Greenhouse/Lever/Ashby/SmartRecruiters/Workday) [1][2][3][9]; jobs.ac.uk RSS [31]; USAJOBS [8].
- **Tier 2 — Breadth (whole web):** SerpApi Google Jobs (worldwide incl. Japan/Asia) [5]; Adzuna (GB/US/DE/FR/AU/NZ/CA/IN/PL/BR/AT/ZA — **no Japan**) [6].
- **Tier 3 — Licensed aggregator:** Active Jobs DB / Fantastic Jobs (LinkedIn/Wellfound/YC + 54 ATS, hourly) [7].
- **Supplemental scraping (optional, behind proxies):** JobSpy for Indeed/Glassdoor (Indeed module reliable, 50+ countries); **avoid its LinkedIn module** [42].

### Visa-sponsorship filtering layer (the deal-breaker gate)
Two complementary layers + a rule engine + a coverage backbone:

1. **Government sponsor registers (employer allow-lists):**
   - **UK** — Register of licensed sponsors, free machine-readable CSV, updated ~every working day; columns include Organisation Name, Town/City, County, "Type & Rating" (A/B), Route. **Verdict-confirmed**, but presence ≠ guaranteed sponsorship for a specific role [22]. The download URL embeds a dated filename — scrape the publication page for the current asset.
   - **Netherlands** — IND public register of recognised sponsors (~12,000+ orgs, monthly); the HSM/kennismigrant route is fast and engineer-friendly. Published as HTML/PDF (scrape) [23].
   - **Canada** — open.canada.ca TFWP Positive LMIA list, quarterly CSV with NOC 2021 codes (filter to 21390 Aerospace, 21301 Mechanical, 21399 Other engineers) [24].
   - **USA** — DOL OFLC disclosure (.xlsx: employer, SOC, prevailing wage, wage level) + USCIS H-1B Employer Data Hub. H-1B is a lottery; his CMAME paper + PhD make **O-1A / EB-1A / EB-2 NIW** far better — these have no public list, so use H-1B/LCA history only as a "visa-willing employer" proxy [25][41].
   - **Australia** — only FOI-released PDFs / commercial directories; weak [26].

2. **Sponsorship-specific boards (role-level self-declared signal — stronger than register-only):**
   - **Arbeitnow API** `?visa_sponsorship=true`, free public JSON, no auth — **verdict-confirmed the param genuinely filters** (DE/AT/CH + remote EU, tech-heavy) [29].
   - **TokyoDev / JapanDev** — hand-curated visa-sponsoring Japan roles; directly relevant to his Tokyo base and existing work visa [30].
   - **SwissDevJobs / Landing.Jobs / Relocate.me** — explicit non-EU sponsorship filters [28].

3. **EU Blue Card rule engine (computed, not a dataset):** As a PhD engineer he clears every 2026 threshold — Germany standard €50,700 / shortage €45,934.20; France €59,373; Spain ~€40-41k. Makes much of the EU effectively "open" [27]. *(Thresholds change yearly and vary by country — refresh annually; verify against official national sources.)*

4. **Coverage backbone (best, not cheapest):** Coresignal / JobsPikr / Bright Data for clean deduplicated worldwide postings, then run registers as enrichment — but these do **not** natively tag sponsorship, so confidence still comes from layers 1-3 [43].

**Gating logic:** boost roles where employer matches a register OR the posting self-declares sponsorship; apply Blue Card eligibility for EU; **hard-exclude** US-citizenship-required/national-lab roles; treat silence as `unknown` (flag for human review, don't auto-reject — avoids dropping good roles where data is sparse outside US/UK).

### Niche CAE/research sources (poorly covered by LinkedIn/Indeed)
- **EU academic/research:** EURAXESS (single highest-fit EU board; **no read API — scrape `/jobs/search`**, confirmed by verdict; explicit visa handling) [44]; jobs.ac.uk RSS [31]; AcademicJobsOnline; Nature Careers (email alerts only); HigherEdJobs RSS.
- **National labs / space agencies:** DLR (German Aerospace Center — sponsor-friendly, international, DAAD/graduate programs) [45]; Space-Careers.com / EuroEngineerJobs; ESA/ONERA/NLR (nationality limits — lower fit). USAJOBS for NASA/US labs (citizenship caveat).
- **CAE-software vendors:** Ansys, Dassault/SIMULIA, Altair, Hexagon/MSC, COMSOL, Siemens DISW. **Two 2025 acquisitions changed the map:** Siemens completed the Altair acquisition (Mar 26, 2025) — Altair roles migrating to jobs.siemens.com, and per the verdict Altair's live ATS is now **Oracle Taleo**, with its SmartRecruiters tenant deprecated/empty [46]. The verdict also found **no evidence Ansys is on SmartRecruiters** (Ansys careers is self-hosted/Phenom; Ansys was acquired by Synopsys, closed Jul 2025). **Re-probe each vendor's live ATS before wiring it up** — do not trust a static vendor→ATS map.
- **Motorsport/aerospace OEM (F1):** Formula Careers, Raceteq; team portals (mostly Workday). UK teams (Mercedes, McLaren, Red Bull, Williams) sponsor visas far more readily than Ferrari/Italian operations — direct fit for his composites/nonlinear-FEM skills [47].

---

## 5. CV/cover-letter tailoring & ATS optimization

**Recommended: DIY Claude (Opus 4.8) pipeline over a verified source-of-truth — not a SaaS resume builder.**

- **Source of truth:** single `resume.yaml` + `professional-experience.yaml` + `cover-letter.yaml`, Pydantic/Zod-validated. The pipeline may only **select and reorder real achievements** — never invent skills, numbers, solver names, or publications. This is the single most important hallucination guardrail, and it matters for visa-relevant claims [10][12].
- **Per-job pass:** Opus 4.8 (1M context fits full CV + long JD + publication list), prompt-cache the master profile (reads ~0.1×), output tailored CV + cover letter + match score + gap report. **Cost ~$0.10/job; ~$50 for 500 jobs**; Batch API is 50% off [verified Claude pricing: Opus 4.8 $5/1M in, $25/1M out].
- **Reference scaffolds to fork:** `javiera-vasquez/claude-code-job-tailor` (3-agent analyze→tailor→edit, weighted requirement scoring, gap analysis, per-company PDF — CC BY-NC, fine for personal use) [11]; or `varunr89/resume-tailoring-skill` (MIT, "truth-preserving optimization" with confidence scores + batch mode for similar roles) [12].
- **ATS validation in the loop:** target **75% min, ~80% sweet spot; never >90%** (that is keyword stuffing and reads as spam) [13]. Use Jobscan (free 5/mo; Premium $49.95/mo per verdict) [14] or **Resume-Matcher** (free, local via Ollama, Apache-2.0, supports Claude via LiteLLM — verdict-confirmed) [34].
- **Two CV variants:** an industry 1-2 pager (for CAE vendors/F1/OEMs) and a longer academic CV with publication list (for labs/academia).
- **ATS formatting rules baked into templates:** reverse-chronological; contact info in the body (not header/footer — ~25% of ATS drop those); Skills section near top; spell out **both** term and abbreviation ("Finite Element Analysis (FEA)", "Isogeometric Analysis (IGA)", "Topology Optimization"); no icons/tables/graphics; .docx or PDF per posting instruction.
- **On AI detection (calibrated):** the strongest evidence is that **AI-*assisted* letters customized with real achievements are viewed favorably** (~63%), and a field study (arXiv 2509.25054) found cover-letter signal value *fell* as AI use spread, shifting employer weight toward verifiable work history — meaning his publications/solvers/quantified results are the differentiator, and **his editing time correlates positively with success** [48]. So human-in-the-loop editing isn't just his rule — it's what makes the output work. Ban boilerplate ("proven track record", "detail-oriented professional", "I am writing to express my interest"). *Caveat: the "X% of recruiters detect AI" survey numbers are mostly vendor-run; treat as directional [50].*

---

## 6. Job-fit matching / ranking

**Recommended: two-stage retrieve-and-rerank + LLM-as-judge with a hand-tuned FEM/CAE rubric.**

- **Stage 1 — bi-encoder retrieval (cheap, over the whole corpus):** embed his profile (one rich "ideal candidate" doc + facet docs for methods/software/domains) and every posting with BGE-large / E5-large / Voyage embeddings into a vector DB; cosine top-N (e.g. top 100-200/day). Sub-$1/day [33].
- **Stage 2 — LLM-as-judge rerank (accurate, only on shortlist):** Claude with a JSON rubric (low temperature, G-Eval style). Sub-scores: `technical_method_fit · software_stack_fit · domain_fit (aero/auto/F1/CAE-vendor) · seniority_fit · research_vs_product_fit`. **Separate gates kept OUT of the blended score:** `visa_sponsorship_signal {yes|likely|unknown|no}`, `location_acceptable`, `language_requirement`. Output: overall 0-100 + sub-scores + one-paragraph rationale + top-3 missing keywords (feeds tailoring). Few cents to low single-digit dollars/day on ~100 items [33].
- **Why not consumer matchers:** Teal's score is the most respected and Jobright's matching is decent, but none expose an API, none have a real worldwide-sponsorship filter (at best US H-1B history), and none are tuned for a PhD FEM niche. Run Teal/Jobright in parallel only as a free cross-check [job-fit, end-to-end dossiers].
- **Reusable component:** `srbhr/Resume-Matcher` as the per-job scorer/gap module (wrap it — it's one-JD-at-a-time, not a batch ranker) [34].
- **Caveat the candidate should know:** "semantic beats keyword by 29-36%" and Resume2Vec "15-16% gains" are vendor/single-paper figures — directionally true that embeddings beat pure keywords, but don't treat the exact percentages as established [verdict].

---

## 7. ToS, legal, account-ban and ethical risks — safe vs risky

This is the most important section and the verdicts are strict here. The candidate is visa-dependent and has a public professional profile; an account ban is a **direct career cost**, not an abstract risk.

### The governing facts (verdict-confirmed)
- **CFAA / hiQ v. LinkedIn:** scraping *public* pages is likely not a CFAA crime, **but hiQ still lost the war** — it accepted a stipulated **$500,000 judgment + permanent injunction** to stop scraping and destroy data. *Verdict nuance:* the $500k covered **five** stipulated claims (breach of contract, CFAA, CA computer-access law, trespass/misappropriation, spoliation sanctions); CFAA rested on **both** data-collection practices **and** fake-account access — not fake accounts alone; and these were stipulated, non-precedential conclusions [15][16][49].
- **LinkedIn User Agreement §8.2** bans bots/scrapers/extensions and automated access; penalty is restriction or permanent termination [17].
- **Indeed ToS** prohibits robots/automated access **and specifically bans automating "Indeed Apply"** — independently validating the no-auto-apply design [end-to-end dossier].
- **Enforcement is real and escalating:** LinkedIn sued Proxycurl, which **shut down permanently Jul 4, 2025** despite ~$10M ARR; HeyReach's company page + founders' profiles permanently banned (Mar 2026); Apollo.io and Seamless.ai banned in 2025 [15][16].
- **GDPR:** keep the pipeline about **jobs (company/role data), not people.** If he ever stores recruiter names/emails, the Art. 14 notice obligation attaches — best avoided by not building dossiers on individuals.

### Safe vs risky

| Action | Risk | Verdict-backed reason |
|---|---|---|
| Consume **public ATS JSON feeds** (Greenhouse/Lever/Ashby/SmartRecruiters) | **Safe** | Intended for programmatic access; no auth, no login, no ToS breach [1][2][3][9]. |
| Use **official APIs** (Adzuna, USAJOBS, SerpApi, Arbeitnow) | **Safe** | Sanctioned programmatic access [5][6][8][29]. |
| Use **licensed aggregator** for LinkedIn data (Active Jobs DB) | **Low** | Vendor assumes the aggregation posture; you never touch LinkedIn. *(Note: their LinkedIn data is polled/aggregated, not formally licensed — vet the vendor's compliance docs.)* [7][verdict]. |
| **Simplify autofill, human clicks Submit** | **Low** | User-click-only is TOS-compliant; the safe submit layer [18]. |
| Scrape **EURAXESS/DLR public search pages** (no login) | **Low-moderate** | Public, no auth; honor robots.txt/ToS; these are irreplaceable visa-friendly sources [44][45]. |
| Scrape **Indeed/Glassdoor via JobSpy behind proxies** | **Moderate** | ToS prohibits robots; Indeed module reliable but is a contract-breach gray zone — supplemental only [42]. |
| Scrape **Workday CXS** at scale | **Moderate-high** | Public/no-auth but ToS bars data-mining; Akamai bot management; needs IP rotation. *Verdict: the 10k cap + "blocked within minutes" specifics are vendor-sourced, not primary* [9]. |
| **Scrape LinkedIn directly** (any account, even public) | **HIGH — do not** | §8.2 breach; hiQ precedent; permanent-ban risk to his network/recruiter visibility [15][17]. |
| **Auto-submit / mass auto-apply** (LazyApply, JobCopilot autopilot, AIHawk LinkedIn Easy Apply) | **HIGH — never** | Violates his rule + Indeed's explicit ban; LinkedIn velocity detection; ~50 Easy Apply/day cap; spray-and-pray underperforms (Indeed's own 2017 "39% less likely" finding — *correctly dated by verdict, not new data*) [20][51]. |

*Down-weighted per verdicts:* LinkedIn ban-rate stats ("23% in 90 days", "340% detection increase", "~40% of tool accounts restricted Q1 2026") are **single-vendor/small-sample marketing estimates** — directionally credible (extensions are the most detectable), but not authoritative [21][52].

**Mitigation summary:** APIs + licensed aggregators for data; public-page scraping only where no API exists and only on no-login pages honoring robots.txt; never automate LinkedIn; keep every Submit a human click; store jobs not people.

---

## 8. Comparison of notable end-to-end / auto-apply products

| Product | What it does | Pricing (verify on vendor site) | Verdict for Yogesh |
|---|---|---|---|
| **Simplify Copilot** | Free autofill across 100+ ATS; user clicks Submit | Core free; Simplify+ ~$39.99/mo ($19.99/wk, $89.99/3mo) [verdict] | **USE (free core)** as the submit layer. Skip Simplify+ — his Claude tailoring is better [18][39]. |
| **Teal** | Tracker + resume builder + match score; never submits | Free tier; Teal+ $9/wk or $29/mo (no annual) [verdict] | **Optional** — best consumer match-score for cross-check + clean tracker [end-to-end]. |
| **Careerflow** | Resume scoring + LinkedIn optimization + tracking | Free; Premium $23.99/mo | **Niche** — LinkedIn-profile polish for recruiter outreach only. |
| **Huntr** | Kanban + AI tailoring + match scoring | Free ≤100 apps; Pro $40/mo | **Optional buy** if he wants the docs/tracker layer off-the-shelf [38]. |
| **Jobright.ai** | AI matching + Turbo auto-apply | Free; Turbo $39.99/mo ($17.99/wk, $89.99/qtr) [verdict] | **Low — US-only** for job location (verdict-confirmed); useful only for visa-sponsored *US* roles via its H-1B filter [end-to-end][verdict]. |
| **JobCopilot** | Auto-apply 500k+ career pages | ~$38.60-55.90/mo, no free trial | **Low** — <2% callback; volume model wrong for senior niche. |
| **Massive** | Auto-apply + manual mode | ~$59/mo | **Low** — off-field matching, US bias. |
| **Adzuna ApplyIQ** | Free agent, ~5 quality apps/wk, auto-submit | Free | **Caution** — still auto-submits sensitive fields; UK/US only; use as discovery aid. |
| **LoopCV** | EU-leaning auto-apply | Free ~10/mo → ~$89.99/mo | **Discovery only** — best EU coverage of the auto tier; keep submit manual. |
| **Scale.jobs** | Human VAs file apps by hand | One-time ~$199/250 apps, ~$299/500 | **Optional safety-net** for a pre-vetted batch; he must vet sponsorship first; loses direct submit control. |
| **LazyApply / Sonara** | Full-auto blast (Sonara shut down Feb 2024) | LazyApply ~$99-249 one-time | **AVOID** — documented wrong H-1B-status fills; ban risk; violates every constraint [20]. |
| **AIHawk (OSS)** | LinkedIn Easy-Apply bot | Free (main repo archived May 2026) | **Reference only** — archived; ban risk. Prefer MadsLorentzen/ai-job-search scaffold [end-to-end]. |
| **RoleStack** | Senior-tech curated shortlist + tailoring | **No public pricing** (vendor self-reported claims, verdict) | **UX reference only** — all claims unverified; verify coverage/pricing before any buy [verdict]. |
| **Active Jobs DB / Fantastic Jobs** | Licensed aggregator (LinkedIn+ATS) | ~$1/1,000 jobs; ~$200-4,000/mo (estimate) [verdict] | **USE as data source** — LinkedIn coverage without scraping risk [7]. |

---

## 9. Phased rollout plan + cost estimate

### Week-by-week
- **Week 1 — Foundation & data.** Write `resume.yaml` (industry + academic variants) with Pydantic schema. Register API keys (SerpApi, Adzuna, USAJOBS, Arbeitnow). Stand up a Postgres/SQLite store + Airtable kanban. **Curate the `{employer → ATS + token}` map** for ~40-60 named targets (Ansys, Dassault, Altair→Siemens/Taleo, Hexagon, COMSOL, F1 teams, DLR, key universities) — re-probe each vendor's live ATS.
- **Week 2 — Backbone search + dedup.** Implement ATS-feed pollers + jobs.ac.uk RSS + SerpApi/Adzuna/Arbeitnow ingestion → normalize + rapidfuzz dedup. Validate volume/precision.
- **Week 3 — Sponsorship gate.** Download UK CSV (nightly), NL IND, Canada LMIA, US OFLC+H-1B; build fuzzy employer matcher + Blue Card rule engine + JD sponsorship-language parser → `sponsorship_signal`. Hard-exclude US-citizenship-only.
- **Week 4 — Match/rank.** Bi-encoder embeddings + Claude LLM-judge rubric; tune on a labeled set of ~30 known good/bad roles. Output sub-scores + gap report.
- **Week 5 — HITL surfacing.** LangGraph `interrupt()` + checkpointer; write top-N to Airtable + 9am Telegram/email digest with evidence pack. Test approve/edit/reject.
- **Week 6 — Tailoring.** Fork claude-code-job-tailor; Opus 4.8 lazy-generation on approve; ATS check via Resume-Matcher/Jobscan; second `interrupt()` for doc review.
- **Week 7 — Assisted apply + track + tune.** Wire Simplify into the manual submit step; close the loop (Applied→Interview→Offer); feed reject reasons back into the rubric. Add EURAXESS/DLR scrapers if EU volume is thin.
- **Ongoing:** weekly rubric tuning; quarterly refresh of LMIA/OFLC and EU Blue Card thresholds; nightly UK CSV.

### Rough monthly cost (best, not cheapest)

| Item | Cost/mo |
|---|---|
| Claude API (matching ~100/day judge + tailoring ~30 approved/mo on Opus 4.8, with caching/batch) | ~$30-80 |
| SerpApi Google Jobs ($75 = 5,000 searches; $25 = 1,000) | $25-75 |
| Active Jobs DB / Fantastic Jobs (licensed LinkedIn coverage, entry tier) | $0-200 (start on free trial / $1-per-1k pay-as-you-go) |
| Adzuna, USAJOBS, Arbeitnow, ATS feeds, jobs.ac.uk RSS | $0 (free) |
| Jobscan Premium (optional; or Resume-Matcher free) | $0-50 |
| Airtable / Simplify core | $0-24 |
| Proxies (only if scraping EURAXESS/Indeed at scale) | $0-30 |
| **Total** | **~$120-260/mo** typical; under $60 if he leans on free tiers + Resume-Matcher |

The dominant "cost" is his engineering time in weeks 1-7, not inference.

---

## 10. Open questions / decisions for the candidate

1. **Geographic priority order?** "Anywhere" is the goal, but ranking (e.g., Japan-stay vs UK vs Germany vs US-via-O-1A) changes which sponsorship layers and niche boards to build first.
2. **US strategy:** pursue O-1A/EB-1A/NIW (strong given CMAME + PhD) rather than the H-1B lottery? This decides how much weight US sources get and whether national labs (citizenship-gated) are worth ingesting at all.
3. **Industry vs academia vs labs split?** Affects CV-variant emphasis and which sources dominate (ATS feeds vs EURAXESS/jobs.ac.uk).
4. **Buy the licensed aggregator now or start free?** Active Jobs DB adds LinkedIn breadth at $0-200/mo — worth it early, or rely on ATS+SerpApi first?
5. **Build depth:** full custom LangGraph core (best fit, weeks of work) vs n8n no-code prototype first to validate the funnel?
6. **Jobscan vs Resume-Matcher** for ATS checking — pay for polished UI + ATS-platform detection, or keep it free/local/private?
7. **Confirm before architecting:** Adzuna/USAJOBS exact rate limits (register to see — *verdict-confirmed undocumented*); Arbeitnow production rate limits; current live ATS for each CAE vendor (Ansys/Altair shifted post-acquisition — *re-probe*); EU Blue Card 2026 thresholds against official national sources.
8. **Networking layer:** LinkedIn is off-limits for automation but valuable for human recruiter outreach — is profile optimization (Careerflow) worth adding alongside the pipeline?

---

## Sources

1. Greenhouse Job Board API — https://developers.greenhouse.io/job-board.html
2. Lever postings-api (GitHub) — https://github.com/lever/postings-api
3. Ashby Public Job Posting API — https://developers.ashbyhq.com/docs/public-job-posting-api
4. 6 ATS Platforms with Public Job Posting APIs (Fantastic Jobs) — https://fantastic.jobs/article/ats-with-api
5. SerpApi Google Jobs API — https://serpapi.com/google-jobs-api
6. Adzuna API Developer Portal — https://developer.adzuna.com/
7. Active Jobs DB — RapidAPI (Fantastic Jobs) — https://rapidapi.com/fantastic-jobs-fantastic-jobs-default/api/active-jobs-db
8. USAJOBS Developer API — https://developer.usajobs.gov/
9. Workday API guide — public CXS endpoint — https://jobspipe.dev/blog/workday-api-guide
10. Claude API model pricing (Anthropic) — https://platform.claude.com/docs/en/pricing
11. claude-code-job-tailor (GitHub) — https://github.com/javiera-vasquez/claude-code-job-tailor
12. resume-tailoring-skill (varunr89, GitHub) — https://github.com/varunr89/resume-tailoring-skill
13. What Jobscan Match Rate Should I Aim For? — https://www.jobscan.co/blog/what-jobscan-match-rate-should-i-aim-for/
14. Jobscan ATS checker — pricing & features — https://www.jobscan.co/
15. hiQ Labs v. LinkedIn — Wikipedia — https://en.wikipedia.org/wiki/HiQ_Labs_v._LinkedIn
16. LinkedIn's Data Scraping Battle with hiQ Labs Ends with Proposed Judgment (Privacy World) — https://www.privacyworld.blog/2022/12/linkedins-data-scraping-battle-with-hiq-labs-ends-with-proposed-judgment/
17. User Agreement — LinkedIn — https://www.linkedin.com/legal/user-agreement
18. Simplify Copilot (official) — https://simplify.jobs/copilot
19. Auto-Apply Job Bots Might Feel Smart — But They're Killing Your Chances (The Interview Guys) — https://blog.theinterviewguys.com/auto-apply-job-bots-might-feel-smart-but-theyre-killing-your-chances/
20. LazyApply Review 2026 (ApplyGhost) — https://applyghost.com/blog/lazyapply-review
21. LinkedIn automation ban risk 2026 (Growleads) — https://growleads.io/blog/linkedin-automation-ban-risk-2026-safe-use/
22. Register of licensed sponsors: workers — GOV.UK — https://www.gov.uk/government/publications/register-of-licensed-sponsors-workers
23. Public register recognised sponsors (Regular labour & highly skilled migrants) — IND Netherlands — https://ind.nl/en/public-register-recognised-sponsors/public-register-regular-labour-and-highly-skilled-migrants
24. TFWP Positive LMIA Employers List — Open Government Portal (Canada) — https://open.canada.ca/data/en/dataset/90fed587-1364-4f33-a9ee-208181dc0b97
25. Performance Data (OFLC disclosure files) — U.S. Department of Labor — https://www.dol.gov/agencies/eta/foreign-labor/performance
26. Accredited sponsor — Australia Dept of Home Affairs — https://immi.homeaffairs.gov.au/visas/employing-and-sponsoring-someone/sponsoring-workers/becoming-a-sponsor/accredited-sponsor
27. EU Blue Card — Make it in Germany — https://www.make-it-in-germany.com/en/visa-residence/types/eu-blue-card
28. tech-jobs-with-relocation — GitHub (AndrewStetsenko) — https://github.com/AndrewStetsenko/tech-jobs-with-relocation
29. Arbeitnow Job Board API — https://www.arbeitnow.com/blog/job-board-api
30. Jobs in Japan with Visa Sponsorship — Japan Dev — https://japan-dev.com/companies/tags/sponsors-visas
31. jobs.ac.uk RSS Feeds — Engineering and Technology — https://www.jobs.ac.uk/feeds/subject-areas/engineering-and-technology
32. USAJOBS Developer API Reference — https://developer.usajobs.gov/api-reference/
33. Cross-Encoders — Sentence Transformers documentation — https://www.sbert.net/examples/cross_encoder/applications/README.html
34. srbhr/Resume-Matcher (GitHub) — https://github.com/srbhr/Resume-Matcher
35. Interrupts — LangGraph Docs (LangChain) — https://docs.langchain.com/oss/python/langgraph/interrupts
36. Automate job search with Claude AI + Google Sheets — n8n workflow template — https://n8n.io/workflows/13584-automate-linkedin-job-search-and-applications-with-claude-ai-and-google-sheets/
37. How to Reduce Notification Fatigue (Courier) — https://www.courier.com/blog/how-to-reduce-notification-fatigue-7-proven-product-strategies-for-saas
38. Huntr Pricing Plans — https://huntr.co/pricing
39. Simplify Copilot Review 2026 (ResumeHog) — https://resumehog.com/blog/posts/simplify-copilot-review-2026-is-the-free-autofill-tool-worth-it.html
40. Slack sendAndWait bug — n8n GitHub issue #13144 — https://github.com/n8n-io/n8n/issues/13144
41. H-1B Employer Data Hub — USCIS — https://www.uscis.gov/tools/reports-and-studies/h-1b-employer-data-hub
42. JobSpy (speedyapply/JobSpy) — GitHub — https://github.com/speedyapply/JobSpy
43. Coresignal Jobs Data API — https://coresignal.com/solutions/jobs-data-api/
44. EURAXESS Jobs Search — https://euraxess.ec.europa.eu/jobs/search
45. DLR Careers — https://www.dlr.de/en/careers
46. Siemens Completes Acquisition of Altair (27 Mar 2025) — https://press.siemens.com/global/en/pressrelease/siemens-acquires-altair-create-most-complete-ai-powered-portfolio-industrial-software
47. Formula Careers — F1 jobs — https://formulacareers.com/jobs-in-f1/
48. Signaling in the Age of AI: Evidence from Cover Letters (arXiv 2509.25054) — https://arxiv.org/abs/2509.25054
49. hiQ and LinkedIn Reach Proposed Settlement in Landmark Scraping Case (Proskauer) — https://www.proskauer.com/blog/hiq-and-linkedin-reach-proposed-settlement-in-landmark-scraping-case
50. 2025 AI in Hiring Report (Insight Global) — https://insightglobal.com/2025-ai-in-hiring-report/
51. Quality, Not Quantity: Why Employers Prefer Targeted Job Applications (Indeed, orig. 2017) — https://www.indeed.com/lead/why-employers-prefer-targeted-job-applications
52. LinkedIn Transparency / Community Report — https://about.linkedin.com/transparency/community-report