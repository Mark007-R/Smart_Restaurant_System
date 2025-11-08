# 🍽️ Smart Restaurant Review Analysis & Business Insights System

An **AI-powered Flask web application** that analyzes restaurant reviews to uncover customer sentiments, detect complaints, and generate actionable insights for business improvement.  
The system integrates data from multiple sources (Zomato, Google, MumbaiRes, etc.) and provides **interactive dashboards** along with a **RAG-based chatbot** for conversational insights.

---

## 🚀 Features

### 🔍 Review Analysis
- Performs **sentiment analysis** (Positive / Negative / Neutral) on restaurant reviews.  
- Identifies **key themes and keywords** in customer feedback.  
- Categorizes reviews into **complaint types** (e.g., Service, Food Quality, Ambience, Pricing).

### 🧠 Business Insights
- Aggregates review data to produce **branch-wise performance comparisons**.  
- Generates **data-driven recommendations** for improving operations.  
- Displays insights through **interactive graphs and charts**.

### 💬 RAG Chatbot (Conversational Insights)
- Ask natural language questions like:  
  > “What are the most common complaints about Café XYZ?”  
  > “Which branch has the best service reviews?”  
- Retrieves context-aware answers from review data.

### 📊 Multi-Source Data Integration
- Accepts **CSV uploads** and **live scraping** from review platforms.  
- Supports filtering, searching, and summarizing by restaurant name, city, or date range.

---

## 🛠️ Tech Stack

| Layer | Technologies Used |
|-------|--------------------|
| **Frontend** | HTML, CSS, Bootstrap |
| **Backend** | Flask (Python) |
| **Database** | SQLite |
| **Data Processing** | Pandas, Requests, AST |
| **Machine Learning / NLP** | Custom sentiment and keyword analysis in `analyzer.py` |
| **Conversational System** | Retrieval-Augmented Generation (`rag_chat.py`) |
| **Visualization** | Matplotlib / Plotly (optional integration) |

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mark007-R/Smart-Restaurant-System.git
   cd Smart-Restaurant-System

2. **Create a virtual environment**
python -m venv venv
source venv/Scripts/activate  # For Windows: venv\Scripts\activate

3. **Install dependencies**
pip install -r requirements.txt

4. **Run the Flask app**
python app.py

5. **Access in your browser**
http://127.0.0.1:5000

👨‍💻 Author
Mark Rodrigues
📍 St. Francis Institute of Technology
💼 Aspiring Full Stack Developer & Data Scientist
🔗 GitHub Profile