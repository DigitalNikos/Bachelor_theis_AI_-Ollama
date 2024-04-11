from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import WebBaseLoader


class Rag:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, model_name="mistral"):
        self.model = ChatOllama(model=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] <<SYS>> You are an assistant for answering questions. Use the following context clues to answer the question. 
            If you don't know the answer, simply say that you don't know. 
            Use a maximum of three sentences and be concise in your response.<</SYS>> [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str, file_extension: str):
        
        docs = PyPDFLoader(file_path=pdf_file_path).load()        

        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        for doc in chunks:
            # Check if 'metadata' exists and is a dictionary
            if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                # Add or update the title in the metadata
                doc.metadata['title'] = "Malakia"

        self.vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )
        
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
        
    def ingest_from_url(self, url: str):
        loader = WebBaseLoader(url)
        docs = loader.load()

        chunks = self.text_splitter.split_documents(docs)

        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                  | self.prompt
                  | self.model
                  | StrOutputParser())

    def ask(self, query: str):
        docs = self.vector_store.similarity_search(query)[0]
        print(f"DOCS: {docs}")
        # print(f"Docs  ::::::::: {docs[0].page_content}")
        
        # context = " ".join(self.conversation_history[-5:]) 
        # combined_query = f"{context} {query}"
        # print(f"Ask context: {context}")
        # messages = [
        #     HumanMessage(
        #         content=combined_query
        #     )
        # ]

        context = docs.page_content
        combined_query = f"{context} {query}"
        messages = [
            HumanMessage(
                content=combined_query
            )
        ]
        print(f"retriever: {self.retriever.get_relevant_documents(query)}")
        if not self.chain:
            answer = self.model.invoke(messages).content
            print(f"Answer not in chain: {answer}")
            return answer
        answer = self.model.invoke(combined_query).content
        print(f"Answer: {answer}")
        return answer
    
    def update_model(self, model_name):
        """
        Updates the model used by the Rag instance.

        Parameters:
        - model_name: str - The name of the new model to use.
        """
        self.model = ChatOllama(model=model_name)
        self.conversation_history = []

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.conversation_history = [] 