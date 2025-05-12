import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

nltk.download('stopwords')

df = pd.read_csv("spam.csv", sep='\t', header=None, names=['label', 'message'])

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['processed'] = df['message'].apply(preprocess)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed'])
y = df['label'].map({ 'spam': 1}) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)


def predict_email(subject, text):
    full_text = f"Subject: {subject}\nMessage: {text}"
    processed = preprocess(full_text)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)
    return "Spam" if prediction[0] == 1 else "Not Spam"



def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def terminal_gui():
    while True:
        clear_screen()
        print(f"{RED}======================================={RESET}")
        print(f"{RED} Welcome to the H@r!n1 Spam Classifier! {RESET}")
        print(f"{RED}======================================={RESET}")
        print(f"{CYAN}1. Classify a custom email message{RESET}")
        print(f"{CYAN}2. View model performance (Accuracy and Report){RESET}")
        print(f"{CYAN}3. Show sample data (Spam) and Ham{RESET}")
        print(f"{CYAN}4. Clear the screen{RESET}")
        print(f"{YELLOW}5. Exit{RESET}")
        
        try:
            choice = int(input(f"\n{GREEN}Enter your choice (1/2/3/4/5): {RESET}"))
            
            if choice == 1:
                subject = input(f"\n{CYAN}Enter the email subject: {RESET}")
                user_input = input(f"\n{CYAN}Enter the email message to classify: {RESET}")
                print(f"{GREEN}Prediction: {predict_email(subject, user_input)}{RESET}")
                input(f"\n{YELLOW}Press Enter to return to the main menu...{RESET}")

            elif choice == 2:
                print(f"{GREEN}Accuracy: {accuracy}{RESET}")
                print(f"{GREEN}\nClassification Report:\n{classification_report_str}{RESET}")
                input(f"\n{YELLOW}Press Enter to return to the main menu...{RESET}")

            elif choice == 3:
                print(f"\n{CYAN}Sample Spam and Ham Messages:{RESET}")
                print(f"\n{RED}Spam Samples:{RESET}")
                print(df[df['label'] == 'spam'].head(5)['message'])
                

                input(f"\n{YELLOW}Press Enter to return to the main menu...{RESET}")

            elif choice == 4:
                clear_screen()

            elif choice == 5:
                quit_choice = input(f"\n{RED}Are you sure you want to exit? (y/n): {RESET}")
                if quit_choice.lower() == 'y':
                    print(f"{MAGENTA}Exiting the H@r!n1 Spam Classifier. Goodbye!{RESET}")
                    break
                else:
                    continue
            else:
                print(f"{RED}Invalid choice. Please try again.{RESET}")
        
        except ValueError:
            print(f"{RED}Please enter a valid number.{RESET}")


if __name__ == "__main__":
    terminal_gui()
