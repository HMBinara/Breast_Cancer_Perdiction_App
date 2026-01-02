Markdown

# 🎀 Breast Cancer Prediction App

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
</div>

---

### 🔍 Project Overview
මෙම යෙදුම (App) මගින් රෝගියෙකුගේ සෛල සාම්පලවල ඇති ලක්ෂණ (features) පරීක්ෂා කර, එම පිළිකාව **Malignant** (අන්තරාදායක) ද නැතහොත් **Benign** (අන්තරාදායක නොවන) ද යන්න නිවැරදිව අනාවැකි පල කරයි.

මෙය **Machine Learning (Logistic Regression)** තාක්ෂණය සහ **Streamlit** framework එක භාවිතයෙන් නිර්මාණය කර ඇත.

### 📊 Dataset Information
මම මේ සඳහා **UCI Machine Learning Repository** හි ඇති **Breast Cancer Wisconsin (Diagnostic) Dataset** එක භාවිතා කළා.
* **Instances:** 569
* **Features:** 30 (Mean Radius, Texture, Perimeter, Area, etc.)
* **Accuracy:** ~95% - 98% (Model performance)

### 🚀 Key Features
- **User-friendly Interface:** ඕනෑම කෙනෙකුට පහසුවෙන් දත්ත ඇතුළත් කළ හැකි UI එකක්.
- **Real-time Prediction:** දත්ත ඇතුළත් කළ සැනින් ප්‍රතිඵලය ලබා දීම.
- **Visual Analytics:** සෛලවල ලක්ෂණ එකිනෙක සැසඳිය හැකි අයුරින් සකසා ඇත.

### 🛠️ How to Run Locally

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/HMBinara/Breast_Cancer_Perdiction_App.git](https://github.com/HMBinara/Breast_Cancer_Perdiction_App.git)
   cd Breast_Cancer_Perdiction_App
Install Dependencies:

Bash

pip install -r requirements.txt
Run the App:

Bash

streamlit run app.py
📁 File Structure
app.py: ප්‍රධාන Streamlit කේතය (Frontend & logic).

model.py / pickle file: පුහුණු කරන ලද ML මොඩලය.

data.csv: පාවිච්චි කරන ලද දත්ත කට්ටලය.

👨‍💻 Developed By
Binara Nethranjana Adhikari

LinkedIn: In/binara-nethranjana-adhikari

GitHub: @HMBinara
