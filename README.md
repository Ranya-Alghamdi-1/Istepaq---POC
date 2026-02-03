# Istepaq – Judicial Attention Monitoring POC

An AI-powered Proof of Concept designed to support judicial integrity in digital court sessions by **monitoring procedural violations and failures in real-time—specifically focusing on participant distraction and gaze deviation.**

---

### 🎯 Objective
Digital courtrooms require strict procedural discipline. **Istepaq** acts as a virtual judicial assistant for the **Judge** by monitoring participant attentiveness through automated visual analysis.

The system ensures procedural compliance by:
* **Real-time Detection:** Identifying when a participant looks away from the camera beyond a defined threshold (detecting visual distraction).
* **Instant Alerting:** Generating visual alerts during the session to notify the Judge of potential procedural failures or misconduct.
* **Automated Logging:** Counting behavioral violations and logging their duration precisely.
* **Judicial Reporting:** Producing a structured JSON summary report for post-session review and documentation.

---

### 🧠 System Workflow

1. **Detection:** Facial and iris landmarks are tracked using `MediaPipe FaceMesh`.
2. **Estimation:** Gaze direction is calculated using specialized rule-based logic.
3. **Validation:** A timer is activated when the participant’s gaze leaves the focal zone.
4. **Action (If threshold exceeded):**
    * Triggers a real-time visual violation alert for the Judge.
    * Increments the violation counter.
    * Logs the specific duration of the distraction.
    * Generates a structured **JSON report** at the end of the session.

---

### ⚙️ Technologies Used
* **Language:** Python
* **Computer Vision:** OpenCV, MediaPipe FaceMesh
* **Algorithm:** Rule-based gaze estimation logic

---

### 📂 Project Structure
```text
Istepaq/
├── src/
│   ├── main.py        # Application entry point
│   ├── gaze.py        # Gaze tracking logic
│   ├── rules.py       # Threshold and violation rules
│   ├── utils.py       # Helper utilities
│   └── config.py      # System configuration
├── data/              # Input media
├── outputs/           # Processed video and JSON reports
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
```

---
📊 Example JSON Output
JSON
{
  "project": "Istepaq Monitor POC",
  "subject": "Participant_01",
  "threshold_no_look_seconds": 2.0,
  "violation_count": 3,
  "violation_total_seconds": 7.4
}

👥 Team – Istepaq

- Dr. Maha Alamri
- Majd Alziyady [[LinkedIn](https://www.linkedin.com/in/majd-alziyady/)]
- Raniyah Alghamdi [[LinkedIn](https://www.linkedin.com/in/raniyah-alghamdi/)]
- Shaimaa Alghamdi [[LinkedIn](https://www.linkedin.com/in/shaimaa-alghamdi/)]
- Maha Al sehly [[LinkedIn](https://www.linkedin.com/in/maha-al-sehly-816355178?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)]


[!IMPORTANT] Note: This project is a technical Proof of Concept developed for the Judicial Intelligence Hackathon. It demonstrates the feasibility of monitoring and detecting violations during digital court proceedings and is intended for research and demonstration purposes only.
