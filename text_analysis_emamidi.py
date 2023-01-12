from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# Importing the StringIO module.
from io import StringIO 
import nltk
import nltk
nltk.download('punkt')
from collections import Counter
import seaborn as sns


st.write(" Analysing Shakespheare's Texts")
st.sidebar.header("Choose your settings")

max_words = st.sidebar.selectbox("Max Words", (20, 50, 70 , 100))
size= st.sidebar.selectbox("Size of words", (400, 500, 600, 700))
width= st.sidebar.selectbox("Select Width", (300, 400, 800, 1000))
word_count = st.sidebar.slider("Minimun Count of Words",5,100,40)

word_stopper= st.sidebar.checkbox("Want to remove Stopwords?" ,value=True)

#image = st.file_uploader("Choose a txt file")
books = {" ":" ","A Mid Summer Night's Dream":"data/summer.txt",
         "The Merchant of Venice":"data/merchant.txt","Romeo and Juliet":"data/romeo.txt"}

final = st.selectbox("Choose a txt file",books)
image = books[final]

if image is not " ":
    raw_text=open(image, "r").read().lower()
    with open(image) as file:
        data= file.read()

stopwords = set(STOPWORDS)
stopwords.update(['us', 'one', 'will', 'said', 'now', 'well', 'man', 'may',
    'little', 'say', 'must', 'way', 'long', 'yet', 'mean',
    'put', 'seem', 'asked', 'made', 'half', 'much',
    'certainly', 'might', 'came','o'])

tab1, tab2, tab3= st.tabs(['File', 'Word Cloud', 'Bar Chart'])

with tab1:
    if image is not " ":
        st.write(data)
with tab2:
    if image is not " ":
        if word_stopper:
            wordcloud= WordCloud(background_color='white',
            max_words=max_words,
            
            width= width)
        else:
            wordcloud= WordCloud(background_color='white',
            max_words=max_words,
            
            width= width)
        wc = wordcloud.generate(data)
        word_cloud = wordcloud.to_file('wordcloud.png')
        st.image(wc.to_array(), width = size)
with tab3:
    if image is not " ":
        
        st.write('Bar chart')
            
        tokens = nltk.word_tokenize(data)
        tokens = [t for t in tokens if t.isalpha()]
        sw_remove = [w for w in tokens if not w.lower() in stopwords]
        if word_stopper:
            frequency = nltk.FreqDist(sw_remove)
            freq_df = pd.DataFrame(frequency.items(),columns=['word','count'])
            sorted_data = freq_df.sort_values("count", ascending=False)
            df = sorted_data[ sorted_data.iloc[:,1]>= word_count ]
            bars = alt.Chart(df).mark_bar().encode(
                x='count',
                y=alt.Y('word:N', sort='-x')
            )
            st.altair_chart(bars, use_container_width=True)
        
        else:
            frequency = nltk.FreqDist(tokens)
            freq_df = pd.DataFrame(frequency.items(),columns=['word','count'])
            sorted_data = freq_df.sort_values("count", ascending=False)
            df = sorted_data[ sorted_data.iloc[:,1]>= word_count ]
             
            bars = alt.Chart(df).mark_bar().encode(
                x='count',
                y=alt.Y('word:N', sort='-x')
            )
            st.altair_chart(bars, use_container_width=True)
        
