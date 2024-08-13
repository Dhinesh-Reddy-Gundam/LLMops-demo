import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
import json
import pandas as pd
from datetime import datetime

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBPw5Tg2qOUcHYv0wUx2F6fRVHz-tfbzSI"

# Initialize Gemini Pro LLM
llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.7)

# Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question: {question}"
)

# Create the LLMChain
chain = LLMChain(llm=llm, prompt=prompt_template)

# Simplified test case and metric classes
class LLMTestCase:
    def __init__(self, input, actual_output, context):
        self.input = input
        self.actual_output = actual_output
        self.context = context

class HallucinationMetric:
    def measure(self, test_case):
        # Simplified hallucination metric (random score for demonstration)
        return round(len(test_case.actual_output) / (len(test_case.input) + 1), 2)

class AnswerRelevancyMetric:
    def measure(self, test_case):
        # Simplified relevancy metric (random score for demonstration)
        return round(len(set(test_case.input.split()) & set(test_case.actual_output.split())) / len(set(test_case.input.split())), 2)

# Function to log results to JSON
def log_to_json(log_entry, filename="qa_log.json"):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
    else:
        data = []
    
    data.append(log_entry)
    
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

# Streamlit app
st.title("Q&A")

# User input
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if question:
        # Generate answer
        answer = chain.run(question)
        
        # Display answer
        st.write("Answer:", answer)
        
        # Create test case
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            context=""  # We don't have context in this simple example
        )
        
        # Evaluate answer
        hallucination_metric = HallucinationMetric()
        hallucination_score = hallucination_metric.measure(test_case)
        
        relevancy_metric = AnswerRelevancyMetric()
        relevancy_score = relevancy_metric.measure(test_case)
        
        # Display evaluation results
        st.write(f"Hallucination Score: {hallucination_score}")
        st.write(f"Relevancy Score: {relevancy_score}")
        
        # Log details
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt_template.template,  # Include the current prompt
            "question": question,
            "answer": answer,
            "hallucination_score": hallucination_score,
            "relevancy_score": relevancy_score
        }
        log_to_json(log_entry)
        
        st.success("Results logged to qa_log.json")
    else:
        st.write("Please enter a question.")

# Optional: Add some information about the app
st.sidebar.header("About")
st.sidebar.info("This is a Q&A application with simplified evaluation metrics.")   