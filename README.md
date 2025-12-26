# AI Job Market Analysis & Salary Intelligence

Scalable NLP + Spark pipeline to analyze technical job skills, demand patterns, and salary drivers from large-scale job posting data.

---

## 📌 Project Overview

This project analyzes job postings at scale to extract **technical skills**, identify **market demand trends**, discover **skill bundles**, and study how skills relate to **salary outcomes**.

The pipeline combines **Natural Language Processing (spaCy)** with **distributed analytics (Apache Spark)** to efficiently process and analyze thousands of job listings.

---

## 🧠 Key Questions Answered

- What technical skills are most in demand?
- How do skill requirements vary by experience level?
- Which skills commonly appear together in job postings?
- Which skill combinations are associated with higher salaries?

---

## 🛠 Tech Stack

- **Python**
- **spaCy** – NLP skill extraction
- **Apache Spark (PySpark)** – scalable aggregation and analytics
- **Pandas** – preprocessing and validation
- **Parquet** – analytics-optimized storage format

---

## 📂 Dataset

The dataset consists of structured job postings including:

- Job title, description, company, and location
- Normalized salary (when available)
- Work type (full-time, part-time, contract, internship)
- Experience level (when available)

---

## 🧩 Pipeline Architecture

Raw Job Postings (CSV)
↓
Data Cleaning (Pandas)
↓
Skill Extraction (spaCy)
↓
Skill Normalization & Explosion (Spark)
↓
Skill Demand Analysis
↓
Skill Co-occurrence (Bundles)
↓
Salary & Skill Insights

---

## 🧹 1. Data Cleaning

- Removed invalid or incomplete records
- Normalized salary values
- Deduplicated job postings by `job_id`
- Prepared text fields for NLP processing

---

## 🧠 2. Skill Extraction (spaCy)

- Built a curated **technical skill lexicon**
- Used `PhraseMatcher` to extract **atomic skills**
- Deduplicated skills per job using Python `set`

**Output format:**
job_id | extracted_skills


> Skills are extracted as individual entities. Skill combinations are discovered later through analysis.

---

## ⚡ 3. Spark Skill Processing

- Loaded NLP output into Spark
- Split and exploded skills into long format:
job_id | skill

- Removed null or empty skills
- Enforced uniqueness defensively

This structure enables scalable aggregation.

---

## 📊 4. Skill Demand Analysis

Computed:
- Top technical skills overall
- Skill demand by experience level
- Distinct job counts per skill (not raw row counts)

---

## 🔗 5. Skill Co-occurrence (Skill Bundles)

Identified skills that frequently appear together within the same job posting.

**Method:**
1. Group skills by job
2. Generate unique skill pairs per job
3. Count co-occurrence across jobs

**Example output:**
skill_a | skill_b | job_count
python | sql | 15432
aws | docker | 11221
spark | aws | 10987

---

## 💰 6. Skill Bundles & Salary Analysis

- Joined skill bundles with salary data
- Computed average salary per skill pair
- Applied minimum job-count thresholds to reduce noise

**Insight example:**
> Skill bundles involving cloud and distributed systems tend to have higher average salaries.

---

## 💾 Outputs

Results are stored in Spark-compatible **Parquet** format:

- `skill_bundles_top.parquet`
- `skill_bundles_salary.parquet`

> Each Parquet output is a directory containing partitioned files (`part-*.parquet`).

---

## 🧠 Design Decisions

- **spaCy before Spark**: NLP is performed once, not distributed
- **Atomic skills only**: avoids combinatorial explosion
- **Distinct job counting**: prevents inflated demand metrics
- **Parquet storage**: optimized for analytics and ML workflows

---

## 🚀 Future Work

- Salary prediction using machine learning
- Feature importance analysis (skills driving pay)
- Role classification (SWE vs Data Engineer vs Analyst)
- Interactive dashboards (Power BI / Tableau)

