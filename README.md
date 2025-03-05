# 🏡 House Price Prediction Model  

**An AI-powered regression model that predicts house prices based on key real estate features.**  
Built using **XGBoost Regressor**, the model provides high-accuracy predictions with an interactive **Gradio UI** for real-time price estimation.

![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-green)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)

---

## **🔍 Project Overview**
- Predicts **house prices** based on **Median Income, Population, House Age**, and other features.
- Uses **XGBoost Regressor**, a powerful machine learning model for regression tasks.
- **Feature Scaling & Hyperparameter Optimization** applied for improved accuracy.
- **Gradio UI** for user-friendly real-time predictions.

---

## **📁 Dataset**
The model is trained on the **California Housing Dataset**, containing:
- **20,640 samples** with **8 key features**.
- Target variable: **House Price (in $1000s).**
- Features include **Median Income, House Age, Population, Average Rooms, Latitude, and Longitude.**

---

## **🛠️ Installation**
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/VinayMattapalli/House-Price-Prediction-ML-Model.git
cd House-Price-Prediction-Model
2️⃣ Install Dependencies
sh
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the Application
sh
Copy
Edit
python House_Predic_model.py
Then open http://127.0.0.1:7860 in your browser.

📊 Model Performance
Metric	Score
Training R² Score	0.8676
Test R² Score	0.8230
Mean Absolute Error	$32,412
📌 The model effectively predicts house prices with an 82.3% accuracy on unseen data.

🖥️ Usage
🎯 Using the Model via Gradio UI
Run:
sh
Copy
Edit
python House_Predic_model.py
Open http://127.0.0.1:7860 in your browser.
Enter house details (e.g., Median Income, House Age, etc.).
Click "Submit" to get the estimated price.
🔗 Technologies Used
Python 3.8+
XGBoost Regressor
Scikit-Learn
Pandas & NumPy
Gradio UI
📝 License
This project is licensed under the MIT License. Feel free to modify and use it.

📬 Contact
👨‍💻 Developed by: Vinay Mattapalli
📧 Email: mvinay2025@gmail.com
🔗 GitHub: VinayMattapalli

🙌 Contributions & feedback are welcome! If you find issues or want to improve the model, feel free to create a pull request.

🚀 Star ⭐ the Repository if You Like It!
If this project helps you, consider giving it a ⭐ on GitHub!

Happy Coding! 🎯🔥

