from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from g4f.client import Client
import random

app = Flask(__name__)

CORS(app, resources={r"/chatbot": {"origins": "http://localhost:3000"}})

questions_reponses = {
    "Question1": ["Hello"],
    "response1": ["hi, how can i help you"],

    "Question2": ["Quel âge as-tu?"],
    "response2": ["J'ai 25 ans"],

    "Question3": ["Que fais-tu ce week-end?"],
    "response3": ["Je suis libre toute la semaine"],

    "Question4": ["ton prénom?"],
    "response4": ["Mon prénom est Barhoumi"],

    "Question5": ["bye"],
    "response5": ["C'était sympa de te parler"]
}

def get_chatgpt_response(user_message):
    try:
        client = Client()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_message}],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API ChatGPT : {str(e)}")
        return f"Erreur lors de l'appel à l'API ChatGPT : {str(e)}"

def get_most_similar_response(user_message, questions_reponses):
    try:
        vectorizer = TfidfVectorizer()

        all_questions = [questions_reponses[key] for key in questions_reponses if key.startswith("Question")]

        flat_questions = [item for sublist in all_questions for item in sublist]

        vectorizer.fit(flat_questions)
        vectors = vectorizer.transform(flat_questions).toarray()
        user_vector = vectorizer.transform([user_message]).toarray()

        similarities = cosine_similarity(user_vector, vectors).flatten()
        max_similarity_index = np.argmax(similarities)

        if similarities[max_similarity_index] > 0.6:
            response_key = f"response{max_similarity_index + 1}"
            return questions_reponses[response_key]

        return None  
    except Exception as e:
        print(f"Erreur dans le calcul de la similarité : {str(e)}")
        return None

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_message = request.json['message']
        
        response_data = get_most_similar_response(user_message, questions_reponses)

        if response_data:
            answer = random.choice(response_data)
            return jsonify({'response': answer})
        
        answer = get_chatgpt_response(user_message)
        return jsonify({'response': answer})
    except Exception as e:
        print(f"Erreur dans le traitement de la requête : {str(e)}")
        return jsonify({'response': f"Erreur interne du serveur : {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
