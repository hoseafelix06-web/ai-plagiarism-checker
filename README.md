AI-Based Plagiarism Checker

Project Overview

The AI-Based Plagiarism Checker is a simple text similarity detection application developed using Python and Natural Language Processing (NLP).

The application compares two text documents and calculates their textual similarity using Term Frequency-Inverse Document Frequency (TF-IDF) and cosine similarity. It can also identify exact matching sentences between the documents.

Aim

To design and implement an application that helps users identify textual similarities between two documents.

Objectives

1. To develop a simple and user-friendly plagiarism-checking interface.
2. To preprocess text before comparison.
3. To calculate document similarity using TF-IDF and cosine similarity.
4. To identify exact matching sentences.
5. To display comparison results clearly.

Technologies Used

- Python
- Streamlit
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF
- Cosine Similarity
- GitHub

Features

- Enter text directly into the application.
- Upload two plain-text files (.txt).
- Calculate textual similarity as a percentage.
- Identify exact matching sentences.
- Display the results in a simple interface.

Limitations

The current version compares only the two documents provided by the user. It does not search the internet or external academic databases. A high similarity score does not automatically prove plagiarism.

How to Run the Application

Install the required packages:

pip install -r requirements.txt

Run the application:

streamlit run app.py

Conclusion

This project demonstrates how basic NLP techniques and machine-learning tools can be used to develop a text similarity detection application for educational purposes.
