from streamlit_chat import message
import streamlit as st
from langchain.vectorstores.redis import Redis
from langchain import FAISS, LLMChain
from langchain.document_loaders import YoutubeLoader
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()


st.title('Chat With Any Youtube Video 🎥💬')

OPENAI_API_KEY = st.text_input("Enter your OpenAI API key", type="password")
 
if OPENAI_API_KEY:
    # OpenAI API key
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    # Create the embeddings object
    embeddings = OpenAIEmbeddings()


    # *********************** Utils ***********************
    def create_db_from_youtube_video_url(video_url):
        # Load the transcript from the video
        loader = YoutubeLoader.from_youtube_url(video_url)
        transcript = loader.load()

        # Split the transcript into chunks of 1000 characters with 100 characters overlap
        # (overlap means that the last 100 characters of a chunk are the first 100 characters of the next chunk)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.split_documents(transcript)

        # Create the vector database
        db = FAISS.from_documents(documents, embeddings)

        return db


    # Get the answer to the question
    def get_response_from_query(db, query):
        # Search the vector database for the most similar chunks
        documents = db.similarity_search(query, k=4)

        # Get the text of the most similar chunks and concatenate them
        content = " ".join([d.page_content for d in documents])

        # Get the large language model (gpt-3.5-turbo)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)

        # Create the prompt template
        prompt_template = """
            You are a helpful assistant that that can answer questions about youtube videos 
            based on the video's transcript: {documents}
    
            Only use the factual information from the transcript to answer the question.
    
            If you feel like you don't have enough information to answer the question, say "I don't know".
    
            Always when answering, dont mention the word "transcript" say "video" instead.
    
            Your answers should be verbose and detailed
            """

        system_message_prompt = SystemMessagePromptTemplate.from_template(prompt_template)

        user_template = "Answer the following question: {question}"
        user_message_prompt = HumanMessagePromptTemplate.from_template(user_template)

        # Create the chat prompt (the prompt that will be sent to the language model)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, user_message_prompt])

        # Create the chain (that will send the prompt to the language model and return the response)
        chain = LLMChain(llm=llm, prompt=chat_prompt)

        # Get the response from the chain
        response = chain.run(question=query, documents=content)
        response = response.replace("\n", "")

        return response


    def generate_response(query, url):
        # Create the vector database
        db = create_db_from_youtube_video_url(url)

        # Get the response
        response = get_response_from_query(db, query)

        return response


    # *********************** Streamlit App ***********************

    # Get the video url from the user
    video_url = st.text_input("Enter a youtube video url")

    # Storing the chat
    if 'question' not in st.session_state:
        st.session_state['question'] = []

    if 'answer' not in st.session_state:
        st.session_state['answer'] = []

    # Get the question from the user
    question = st.text_input("Enter a question : ")

    if question:
        res = generate_response(question, video_url)
        st.session_state['question'].append(question)
        st.session_state['answer'].append(res)

    if st.session_state['answer']:
        for i in range(len(st.session_state['answer'])):
            message(st.session_state['question'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["answer"][i], key=str(i))