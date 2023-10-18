from flask import Flask, request, jsonify
import spacy
import numpy as np
from dotenv import load_dotenv
import openai
from scipy.spatial.distance import cosine
import os
import psycopg2

app = Flask(__name__)

load_dotenv()
openai.api_key = os.getenv('openai_API_key')
emb_source = 'openai'

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Define a function to calculate the embedding of a text
def calculate_embedding(text, module='spacy'):
    
    if module=='openai':
        model="text-embedding-ada-002"
        text = text.replace("\n", " ")
        embedding = np.array(openai.Embedding.create(input = [text], model=model)['data'][0]['embedding'])
    else:
        nlp = spacy.load("en_core_web_sm")
        embedding = nlp(text).vector
    
    return embedding

# Define a function to calculate the cosine similarity between two vectors
def cosine_similarity(vector1, vector2):
    return 1 - cosine(vector1, vector2)

# Define a function to retrieve the most similar(based on cosine similarity) document based on question embeddings
def find_most_similar_document(question, db_connection):
    cursor = db_connection.cursor()

    # Calculate embeddings for the question and convert to a NumPy array
    question_embedding = calculate_embedding(question, module=emb_source)

    # Query the database to retrieve all document embeddings
    cursor.execute("SELECT sent_text, embedding FROM document_embeddings")
    rows = cursor.fetchall()

    most_similar_sent_text = None
    highest_similarity = -1  # Initialize with a low value

    for row in rows:
        sent_text, embedding_bytes = row
        # Convert the stored bytes to a NumPy array
        doc_embedding = np.frombuffer(embedding_bytes, dtype=question_embedding.dtype)

        # Calculate cosine similarity between question and document embeddings
        similarity = cosine_similarity(question_embedding, doc_embedding)

        # Update if similarity is higher
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_sent_text = sent_text

    cursor.close()

    return most_similar_sent_text

# Define a function to answer a question using ChatGPT
def answer_question(question, document_content):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Answer the following question: {question},\n based on the next context: {document_content}",
        max_tokens=50,  # Adjust max tokens as needed
    )
    return response.choices[0].text

@app.route('/get_answer', methods=['POST'])
def get_answer():
    data = request.get_json()
    question = data.get('question', '')
    
    if question:
        # Connect to your Google Cloud PostgreSQL database
        db_params = {
                    "dbname": os.getenv("dbname"),
                    "user": os.getenv("user"),
                    "password": os.getenv("password"),
                    "host": os.getenv("host"),
                    "port": os.getenv("port"),  
                    }
        connection = psycopg2.connect(**db_params)

        # Find the most similar document based on the question
        context = find_most_similar_document(question, connection)

        if context:
            # Generate an answer using OpenAI's API
            answer = answer_question(question, context)
            connection.close()
            return jsonify({'answer': answer.strip()})
        else:
            connection.close()
            return jsonify({'error': 'Document content not found'})
    else:
        return jsonify({'error': 'Missing question'})
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
