import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model
import json
import random
import os
from flask import Flask, render_template, request
import pickle
import spacy
import numpy as np  

nlp = spacy.load("en_core_web_md")

greetings = [
    "Hello! Are you ready for the Interview?",
    "Ready to start your interview?",
    "Let's begin the interview.",
    "Excited to chat with you. Shall we start the interview?",
    "Hello! It's interview time. Are you ready?"
]

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(script_dir, 'data.json')

with open(data_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

questions = data["questions"]

words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

model = load_model('model.h5')

question_counter = 0

bot_questions = []
user_answers = []

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_next_question(previous_intent, user_response):
    
    relevant_intent = next((intent for intent in questions if intent['tag'] == previous_intent), None)
    dynamic_questions = relevant_intent.get('dynamic_questions', []) if relevant_intent else []
    
    if dynamic_questions:
        return random.choice(dynamic_questions)
    else:
        return random.choice(questions)

def validate_answer(candidate_answer, actual_answer):
    candidate_doc = nlp(candidate_answer)
    actual_doc = nlp(actual_answer)
    similarity = candidate_doc.similarity(actual_doc)
    return similarity

def calculate_percentage_score(similarity_scores):
    interview_score_percentage = (sum(similarity_scores) / len(similarity_scores)) * 100
    return interview_score_percentage

def analyze_sentiment(similarity_scores):
    avg_similarity = sum(similarity_scores) / len(similarity_scores)

    sentiment = "positive" if avg_similarity > 0.6 else "neutral" if avg_similarity > 0.4 else "negative"

    if sentiment == "positive":
        summary = "The candidate demonstrated a strong understanding of the topics, providing clear and detailed answers. Practical knowledge was well exhibited, reflecting a high level of expertise."
    elif sentiment == "neutral":
        summary = "The candidate's responses were generally satisfactory, covering the essential aspects. However, there is room for improvement in providing more detailed and practical insights."
    else:
        summary = "The candidate struggled to convey a solid understanding. Responses lacked clarity and depth, indicating a need for further knowledge and practical experience."

    return sentiment, summary

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def calculate_similarity(user_response, expected_answer):
    preprocessed_user = preprocess_text(user_response)
    preprocessed_expected = preprocess_text(expected_answer)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([preprocessed_user, preprocessed_expected])

    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity

app = Flask(__name__)
app.static_folder = 'static'
similarity_scores = []
selected_questions = []

def select_random_questions():
    global selected_questions
    available_questions = list(questions)
    for _ in range(5):
        selected_question = random.choice(available_questions)
        selected_questions.append(selected_question)
        available_questions.remove(selected_question)

select_random_questions()

@app.route("/")
def home():
    global bot_questions
    global user_answers

    
    bot_questions = []
    user_answers = []

    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    global question_counter
    global selected_questions
    global similarity_scores  
    global bot_questions
    global user_answers

    user_text = request.args.get('msg')
    response = ""

    if question_counter == 0:
        response = random.choice(greetings)
        question_counter += 1  
    elif question_counter <= 5:
        current_question = selected_questions[question_counter - 1]
        response = current_question["question"]
        bot_questions.append(current_question) 

        if question_counter > 1:
            user_answers.append(user_text)  

        actual_answer = current_question["solution"]
        similarity = validate_answer(user_text, actual_answer)
        similarity_scores.append(similarity)

        question_counter += 1
    else:
       
        user_answers.append(user_text)

        interview_score_percentage = (sum(similarity_scores) / len(similarity_scores)) * 100
        score_message = f"\nYour Interview Score: {interview_score_percentage:.2f}%"
        sentiment, summary = analyze_sentiment(similarity_scores)
        response = f"{score_message}"

        save_interview_summary(f"Interview Score: {interview_score_percentage:.2f}%\nSentiment: {sentiment.capitalize()}\nSummary: {summary}", bot_questions, user_answers)

        question_counter = 0
        similarity_scores = []
        selected_questions = []
        bot_questions = []
        user_answers = []
        select_random_questions()

    return response


def save_interview_summary(interview_summary, bot_questions, user_answers):
    file_path = "interview_summary.txt"
    with open(file_path, "w") as file:
        file.write(interview_summary)
        file.write("\n\nQuestions and Answers:\n")
        for bot_q, user_a in zip(bot_questions, user_answers):
            file.write(f"\nBot Question: {bot_q['question']}\nUser Answer: {user_a}\n")

if __name__ == "__main__":
    app.run()
