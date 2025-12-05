Here’s a polished, ready-to-paste **README** for your repo with your screenshot included:

````markdown
# 🥚 Cracked Egg Detection – AI Object Detection App

A complete end-to-end **Egg vs Cracked Egg Detection** application using **YOLO** and **Streamlit**. This project detects **cracked eggs** vs **whole eggs** from images, videos, or live webcam feed and provides a **real-time interactive dashboard** with detection statistics.  

---

## 🚀 Features

- Detects **Cracked Eggs** and **Whole Eggs** in images, videos, and live webcam feed.
- Real-time **Webcam detection** with bounding boxes.
- Adjustable **confidence threshold** and maximum detections per frame.
- **Annotated image/video download**.
- **Object counting** with visual statistics (bar charts).
- User-friendly **Streamlit dashboard** for easy interaction.
- Clean, organized, and commented code for easy understanding.

---

# Cracked Egg Detection Dashboard

<img src="https://raw.githubusercontent.com/maksudrakib44/cracked-egg-detection/master/assets/Dashboard02.png" width="800" alt="Dashboard Screenshot">

**Figure:** Dashboard interface showing real-time egg quality detection results.
---

## 💻 Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/maksudrakib44/cracked-egg-detection.git
cd cracked-egg-detection
````

2. Create a virtual environment (optional but recommended):

```bash
conda create -n crack_egg python=3.10 -y
conda activate crack_egg
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place your trained YOLO model (`egg_eggDetector_best.pt`) in the project root.

5. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 🏗 Project Structure

```
cracked-egg-detection/
│
├── assets/                  # Screenshots and visual assets
├── app.py                   # Streamlit dashboard application
├── egg_eggDetector_best.pt  # Trained YOLO model
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

---

## 🧠 How It Works

1. **Image Detection:** Upload an image of eggs → detects cracked and whole eggs → shows annotated image → download option → counts & chart display.
2. **Video Detection:** Upload a video → YOLO processes frame by frame → annotated video → download → aggregated counts & chart.
3. **Webcam Detection:** Capture live webcam frames → YOLO predicts in real-time → annotated frame display → counts & FPS display.

---

## 📦 Dataset

* Annotated dataset available on [Roboflow Workspace](https://app.roboflow.com/maksud)
* Classes: `cracked_egg`, `whole_egg`
* Dataset includes 300+ images of eggs with bounding boxes.

---

## 📈 Model

* YOLO custom-trained on **Cracked Egg Dataset**.
* Achieved high detection accuracy (mAP ~0.95).
* Trained weights: `egg_eggDetector_best.pt`.

---

## 📝 Notes

* For best results, use a **plain background** during webcam detection.
* Show eggs clearly in frame for accurate detection.
* The confidence slider allows dynamic adjustment for stricter or looser detections.

---

---

## ⚡ Future Improvements

* Add **more egg conditions** (e.g., rotten, half-cracked).
* Integrate **mobile-friendly interface** for on-field detection.
* Export results as **Excel/CSV** for inventory tracking.

---

## 👤 Author

This project is developed by Md. Maksudul Haque as part of the IEEE Computer Society SBC GUB AI/ML Bootcamp.
Feel free to fork, clone, or experiment with the model and dashboard.

* Email: [maksudrakib44@gmail.com](mailto:maksudrakib44@gmail.com)
* GitHub: [maksudrakib44](https://github.com/maksudrakib44)
* Portfolio: [Md. Maksudul Haque](https://maksud-portfolio.vercel.app/)

---

## 📄 License

This project is for educational purposes only. Please do not use for commercial purposes without permission.

