# Spam-email-deduction
**Spam-email-deduction** is a machine learning-based email classification tool that automatically detects whether an email is **spam** or **not spam**. Leveraging **Natural Language Processing (NLP)** techniques and a **Naive Bayes** model, the classifier processes raw email text and outputs predictions with high accuracy.

---

## 🚀 Features

* ✅ Classify custom email messages as spam or ham
* 📊 View performance metrics like accuracy, precision, and recall
* 📂 Explore sample spam and ham emails
* 🧹 Clear terminal screen for a clean interface
* ❌ Exit anytime from the CLI menu

---

## 🧠 Techniques Used

* **Preprocessing**: Tokenization, stemming, stop-word removal
* **Feature Extraction**: TF-IDF Vectorization
* **Model**: Multinomial Naive Bayes
* **Evaluation**: Accuracy, confusion matrix, precision, recall, F1-score

---

## 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/h-spam-classifier.git
   cd Spam-email-deduction
   ```

2. Create and activate virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate     # On Windows
   # OR
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Running the Application

To launch the classifier interface:

```bash
python main.py
```

You’ll be presented with a menu like:

```
==================================
 Welcome to the Spam-email-deduction!
==================================
1. Classify a custom email message
2. View model performance (Accuracy and Report)
3. Show sample data (Spam and Ham)
4. Clear the screen
5. Exit

Enter your choice (1/2/3/4/5):
```

---

## 📁 Project Structure

```
Spam-email-deduction/
│
├── data/                   # Dataset files
├── models/                 # Saved models
├── main.py                 # CLI entry point
├── spam_classifier.py      # Core classification logic
├── preprocessing.py        # NLP utilities
├── requirements.txt        # Required Python packages
└── README.md               # This file
```

---

## 📈 Example Performance

* **Accuracy**: 97%
* **Precision (Spam)**: 95%
* **Recall (Spam)**: 94%
* **F1-Score (Spam)**: 94.5%

---

## 🛠️ Technologies

* Python 3.x
* Scikit-learn
* NLTK
* Pandas
* NumPy

---

## 🙌 Contributing

Contributions are welcome! Feel free to fork this repo and submit pull requests.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


