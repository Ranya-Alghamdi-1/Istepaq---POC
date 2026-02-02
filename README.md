#Istepaq - POC 

AI-powered Proof of Concept designed to support judicial integrity in digital court sessions through automated visual attention monitoring.

🎯 Objective

Digital courtrooms require strict procedural discipline.
Istebaq acts as a virtual judicial assistant by monitoring participant attentiveness in real time.

The system:

Detects when a participant looks away from the camera beyond a defined threshold

Generates instant visual alerts during the session

Counts behavioral violations

Produces a structured JSON summary report for review

This ensures violations are detected precisely and within clearly defined time constraints.

🧠 System Workflow

Detection
Facial and iris landmarks are tracked using MediaPipe FaceMesh.

Estimation
Gaze direction is calculated using rule-based logic.

Validation
A timer is activated when the participant’s gaze leaves the focal zone.

Action (if threshold exceeded)

Trigger real-time visual violation alert

Increment violation counter

Log violation duration

Generate structured JSON report at session end

⚙️ Technologies Used

Python

OpenCV

MediaPipe FaceMesh

Rule-based gaze estimation logic

📂 Project Structure
Istebaq/
├── src/
│   ├── main.py
│   ├── gaze.py
│   ├── rules.py
│   ├── utils.py
│   └── config.py
│
├── data/
├── outputs/
│   ├── istepaq_demo_output.mp4
│   └── summary.json
│
├── requirements.txt
└── README.md

📊 Example JSON Output
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

⚠️ Note

This project is a technical Proof of Concept developed for the Judicial Intelligence Hackathon. It demonstrates feasibility of automated visual attention monitoring and is intended for research and demonstration purposes only.
