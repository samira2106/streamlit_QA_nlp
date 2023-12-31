from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import streamlit as st
import pandas as pd

model_name = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

def main():
  st.title("Question Answering with Transformers")

  # Get user input
  context = st.text_area("Context", height=200)
  question = st.text_area("Question", height=100)

  # Generate predictions
  QA_input = {
      'question': context,
      'context': question
  }
  res = nlp(QA_input)

  # Display predictions
  st.subheader("Predictions")
  st.write(res['answer'])

if __name__ == "__main__":
  main()
