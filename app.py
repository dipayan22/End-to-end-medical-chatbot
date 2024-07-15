from flask import Flask,render_template,jsonify,request
from src.helper import hugging_face_embeddings
# from langchain.vectorstores import Pinecone

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatGooglePalm
from langchain.chains import RetrievalQA
from src.prompt import *
from dotenv import load_dotenv
import os
from pinecone import Pinecone



app=Flask(__name__)

load_dotenv()

embeddings=hugging_face_embeddings()

Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

index_name='medical-chatbot'

from langchain.vectorstores import Pinecone

docsearch=Pinecone.from_existing_index(index_name,embeddings)

PROMPT=PromptTemplate(template=prompt_template,input_variables=['context','question'])
chain_type_kwargs={"prompt": PROMPT}

llm=ChatGooglePalm(google_api_key=os.getenv('GOOGLE_API_KEY'),temperature=0.5)


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs)




@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get",methods=['GET','POST'])
def chat():
    query=request.form['msg']
    result=qa(query)
    return str(result["result"])


if __name__=='__main__':
    app.run(debug=True)
