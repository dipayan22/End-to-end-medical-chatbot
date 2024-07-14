from src.helper import text_split,pdf_loader,hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')

extract_data=pdf_loader('Data/')

text_chunks=text_split(extract_data)

embeddings=hugging_face_embeddings()

pc=Pinecone(api_key=PINECONE_API_KEY)

index_name='medical-chatbot'

#Creating Embeddings for Each of The Text Chunks & storing
docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

