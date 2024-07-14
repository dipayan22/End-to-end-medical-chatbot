from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import HuggingFaceEmbeddings

# extract data from PDF

def pdf_loader(data):
    loader=DirectoryLoader(data,
                        glob='*.pdf',
                        loader_cls=PyPDFLoader)

    document=loader.load()

    return document

# create text chunks

def text_split(extracted_data):
    text_spliter=RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=100)
    text_chunks=text_spliter.split_documents(extracted_data)

    return text_chunks


# dowmload embedded model
def hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings