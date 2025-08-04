# Intelligent Experiment Design Dashboard

> Originally developed as a final project for CMU's Interactive Data Science course (IDS, Spring 2021)

This interactive Streamlit app helps users design new experiments using Bayesian Optimization or Active Learning. By uploading previously evaluated experiments and a candidate pool, users can iteratively explore, visualize, and prioritize next experiments — making data collection more efficient and intelligent.

---

## 🔗 Links

- 🧪 **[Access the app](https://zhiyunapps-active-experiment-dashboard.share.connect.posit.cloud/)**  
- 🎥 **[Video Demonstration](https://drive.google.com/file/d/1fQcvdrkZH0zkAgY54eGMzErAVZXFz-0G/view?usp=sharing)**  
- 📄 **[Final Report](https://github.com/CMU-IDS-2021/fp--zhiyun/blob/main/Report.md)**  

---

## 💻 Application Interface

### 📈 Regression Mode
![image](https://github.com/CMU-IDS-2021/fp--zhiyun/blob/main/imgs/app_regression.png)


## ⚙️ Features

- **Two modes**: regression model training & objective optimization
- **Bayesian Optimization** using Upper Confidence Bound (UCB)
- **Active Learning** based on model uncertainty
- **Interactive PCA visualizations** for candidate suggestions
- **Streamlit UI** with sliders, uploaders, and real-time feedback
- **Easy CSV input/output** for experiments and candidate pools

---

## 🧪 How It Works

1. Upload:
   - Your previously evaluated experiments (`.csv` with input parameters and a column called `Objective`)
   - A pool of unlabeled candidate experiments (same input columns)

2. Choose a mode:
   - **Regression**: Fit a Gaussian Process model and improve it iteratively using uncertainty sampling
   - **Optimization**: Use Bayesian Optimization to find inputs that maximize the objective

3. View and export suggestions for your next experiments

---

## 📦 Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/ZhiyunGong/active-experiment-dashboard.git
cd active-experiment-dashboard
pip install -r requirements.txt
streamlit run main.py
