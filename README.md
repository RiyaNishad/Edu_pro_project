EduPro Course Demand & Revenue Dashboard
A Streamlit-powered predictive analytics dashboard for optimizing online course pricing, instructor allocation, and demand forecasting on the EduPro Online Learning Platform.

[

🎯 Features
Real-time Metrics: Course enrollments, revenue, and instructor performance KPIs

Category Insights: Revenue and enrollment trends by course category

ML Predictions: Random Forest models predict enrollments and revenue for new courses

Interactive Controls: Test pricing strategies and instructor quality impacts

Model Validation: Cross-validated performance metrics with feature importance

📊 Dashboard Tabs
Tab	Purpose	Key Visuals
Overview	Platform KPIs	Revenue bars, enrollment trends
Category Insights	Category performance	Price vs Enrollment scatter
Prediction	New course forecasting	Interactive input sliders
Model Performance	ML model diagnostics	Feature importance ranking
🛠 Tech Stack
text
Core: Streamlit • Pandas • Scikit-learn • Plotly
Data: Excel multi-sheet (Users, Teachers, Courses, Transactions)
ML: Random Forest Regressor (200 estimators)
Deployment: Streamlit Community Cloud
🚀 Live Demo
Open Dashboard

📈 Sample Predictions
Input: Data Science course, $199, 25hrs, 4.2⭐ rating, 6yr avg teacher exp
Output: ~42 enrollments, $8,350 revenue (predicted)

🔧 Local Setup
Clone & Install

bash
git clone <your-repo>
cd edupro-dashboard
pip install -r requirements.txt
Run Locally

bash
streamlit run app.py
Required Data
Place EduPro Online Platform.xlsx in project root (contains: Users, Teachers, Courses, Transactions sheets)

📁 Project Structure
text
├── app.py                 # Main Streamlit dashboard
├── requirements.txt       # Dependencies
├── EduPro Online Platform.xlsx  # Source data
└── README.md             # This file
🤖 ML Model Details
Targets: Enrollment Count, Total Revenue
Features: Price, Duration, Rating, Teacher Experience/Rating, Category/Type/Level dummies
Metrics: MAE, R² scores displayed in Model Performance tab
Validation: 80/20 train-test split

📈 Key Insights Generated
Optimal pricing by course category

High-impact instructor characteristics

Revenue vs enrollment trade-offs

Category performance benchmarking

👩‍💻 Author
Riya Nishad
BSc Data Science - AI/ML Intern Candidate
vadodara, Gujarat | LinkedIn

Skills Demonstrated: Data Engineering, ML Deployment, Interactive Visualization, Cloud Deployment

📄 License
MIT License - Feel free to fork, modify, and use for portfolio/educational purposes.

Built for Data Science internship applications - April 2026
Deployed on Streamlit Community Cloud 🚀
