import torch
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)


st.set_page_config(
    page_title="Symptom Similarity App",
    page_icon="🩺",
    layout="wide",
)


# Load your dataset
df = pd.read_csv('Symptom2Disease.csv')


# Load the GPT-2 model and tokenizer
model_path = "gpt2_model"
model = GPT2Model.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Load the embeddings from the file
embeddings_array = np.load('embeddings.npy')

# Convert the NumPy array back to a torch.Tensor
embeddings_matrix = torch.tensor(embeddings_array)

# Function to generate sentence embeddings
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Function to find top similar sentences
def get_top_similar_sentences(input_sentence, top_k=5):
    input_embedding = get_sentence_embedding(input_sentence)
    similarities = cosine_similarity(
        input_embedding.detach().numpy(), embeddings_matrix.detach().numpy()
    )
    indices = similarities.argsort(axis=1)[:, ::-1][:, 1 : top_k + 1]
    similar_sentences = [df.iloc[i]['text'] for i in indices[0]]
    return similar_sentences

# Function to generate word cloud
def generate_wordcloud():
    wordcloud = WordCloud(width=400, height=400).generate(' '.join(df['text']))
    fig = plt.figure(figsize=(3,3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(fig, figsize=(3,3))

# Function to generate non-interactive plot
def generate_simple_plot():
    symptom_counts = df['label'].value_counts()

    # Plot horizontal bar chart
    plt.figure(figsize=(8, 6))
    symptom_counts.sort_values().plot(kind='barh')
    plt.xlabel('Count')
    plt.ylabel('Symptom')
    plt.title('Symptom Distribution')
    st.pyplot()
    
    
# Streamlit UI
st.title("Symptom Similarity App")

# Divide the screen vertically into two columns
left_column, right_column = st.columns(2)

# Selectbox for predefined samples
predefined_samples = ["Joint pain in fingers", "Skin peeling on elbows", "Fatigue and malaise"]
selected_sample = left_column.selectbox("Select a predefined sample or enter your own:", predefined_samples, index=None)

# User input
user_input = left_column.text_input("Enter a symptom description:")

# Use the selected sample if provided
if selected_sample and not user_input:
    user_input = selected_sample

if user_input:
    with st.spinner("Finding similar sentences..."):
        similar_sentences = get_top_similar_sentences(user_input)
        
    left_column.header("Top 5 Similar Sentences:")
    for i, sentence in enumerate(similar_sentences, 1):
        left_column.write(f"{i}. {sentence}")
        
    if not similar_sentences:
        left_column.warning("No similar sentences found. Please try another input.")


# WordCloud and Interactive Plot on the right column
with right_column:
    right_column.subheader("Word Cloud")
    generate_wordcloud()

    right_column.subheader("Symptom Distribution")
    generate_simple_plot()
