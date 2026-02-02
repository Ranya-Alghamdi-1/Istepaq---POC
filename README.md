# Istepaq – Judicial Attention Monitoring POC

An AI-powered Proof of Concept designed to support judicial integrity in digital court sessions through automated visual attention monitoring.

---

## 🎯 Objective

Digital courtrooms require strict procedural discipline.  
**Istepaq** acts as a virtual judicial assistant for the **Judge** by monitoring participant attentiveness in real time.

The system ensures procedural compliance by:

- **Real-time Detection:** Identifying when a participant looks away from the camera beyond a defined threshold.  
- **Instant Alerting:** Generating visual alerts during the session to notify the Judge of potential procedural failures.  
- **Automated Logging:** Counting behavioral violations and logging their duration precisely.  
- **Judicial Reporting:** Producing a structured JSON summary report for post-session review and documentation.  

---

## 🧠 System Workflow

1. **Detection:** Facial and iris landmarks are tracked using `MediaPipe FaceMesh`.  
2. **Estimation:** Gaze direction is calculated using rule-based logic.  
3. **Validation:** A timer is activated when the participant’s gaze leaves the focal zone.  
4. **Action (If threshold exceeded):**
   - Trigger real-time visual violation alert.  
   - Increment violation counter.  
   - Log violation duration.  
   - Generate structured JSON report at session end.  

---

## ⚙️ Technologies Used

- **Language:** Python  
- **Computer Vision:** OpenCV, MediaPipe FaceMesh  
- **Algorithm:** Rule-based gaze estimation logic  

---

## 📂 Project Structure

```text
Istebaq/
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
## 📊 Example JSON Output
{
  "project": "Istebaq Monitor POC",
  
  "subject": "Participant_01",
  
  "threshold_no_look_seconds": 2.0,
  
  "violation_count": 3,
  
  "violation_total_seconds": 7.4
  
}

👥 Team – Istebaq

Dr. Maha Alamri

Majd Alziyady

Raniyah Alghamdi

Shaimaa Alghamdi

Maha Alsehli

⚠️ Important Note

This project is a technical Proof of Concept developed for the Judicial Intelligence Hackathon.
It demonstrates the feasibility of automated visual attention monitoring and is intended for research and demonstration purposes only.
