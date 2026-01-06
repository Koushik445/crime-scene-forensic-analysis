# From Detection to Deduction  
## Explainable Evidence-Based AI Reasoning for Crime Scene Analysis

This project presents an end-to-end, explainable forensic analysis system that integrates
computer vision and natural language processing to assist in preliminary crime-scene investigations.
The system detects key forensic evidence from images and verifies eyewitness statements using
semantic reasoning, ultimately generating a structured and interpretable forensic report.

---

## 📌 Project Overview
Manual crime-scene analysis is time-consuming and prone to human error, especially when multiple
objects and unreliable eyewitness accounts are involved.  
This project automates early-stage forensic analysis by:

- Detecting critical forensic evidence from crime-scene images using YOLO
- Comparing detected evidence with eyewitness statements using semantic similarity
- Applying rule-based reasoning to identify corroborations, omissions, and contradictions
- Generating an explainable, human-readable forensic report

The system follows the principle **“From Detection to Deduction”** — moving beyond object detection
towards structured forensic reasoning.

---

## 🎯 Objectives
- Detect important forensic objects from crime-scene images
- Analyze eyewitness statements using semantic embeddings
- Compare visual evidence with textual descriptions
- Identify corroborated, missing, and contradictory evidence
- Compute a witness credibility score
- Automatically generate a structured forensic analysis report

---

## 🧠 Evidence Classes Supported
The YOLO model is trained to detect the following forensic categories:
- Blood
- Handgun
- Shotgun
- Knife
- Hammer
- Mobile Phone
- Rope

---

## 🏗️ System Architecture
The system follows a modular pipeline:

1. **Input**
   - Crime-scene images
   - Eyewitness textual statements

2. **Evidence Detection**
   - YOLO-based object detection
   - Bounding boxes, class labels, confidence scores

3. **Text Processing & Semantic Reasoning**
   - Statement cleaning and synonym handling
   - Sentence-level embeddings
   - Similarity-based verification

4. **Rule-Based Forensic Reasoning**
   - Corroboration, omission, contradiction analysis
   - Evidence importance weighting
   - Credibility score estimation

5. **Automated Report Generation**
   - Evidence inventory
   - Consistency analysis
   - Incident interpretation
   - Investigative recommendations

---

## 🛠 Technologies Used
- **Python**
- **YOLOv8 (Ultralytics)**
- **PyTorch**
- **OpenCV / PIL**
- **SentenceTransformers (BERT-based embeddings)**
- **Rule-based reasoning logic**

---

## 📁 Project Structure
crime-scene-forensic-analysis/
│
├── config/
│ ├── cropped_data.yaml
│ └── scene_data.yaml
│
├── pretrain_yolo.py # Phase-1 cropped evidence training
├── finetune_yolo.py # Phase-2 full scene fine-tuning
├── inference_on_test.py # Evidence detection inference
├── inference_with_heatmaps.py # Explainable heatmap visualization
├── forensic_analysis.py # Reasoning & report generation
│
├── .gitignore
└── README.md
