import os
os.environ["OPENAI_API_KEY"] = "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"
os.environ['TRANSFORMERS_CACHE'] = '/tmp'
# https://qwq4uzmkaq6tmmj7afeozfguke0ulend.lambda-url.us-east-1.on.aws/
import json
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory

model_kwargs = {'device': 'cpu'}
default_ef = SentenceTransformerEmbeddings(model_name="./all-MiniLM-L6-v2", model_kwargs=model_kwargs)
print("passing default_ef")

db = FAISS.load_local("./db", default_ef)

print("passing chromadb")
gpt_turbo = ChatOpenAI(temperature=0.3, model_name='gpt-3.5-turbo-16k')
print("passing gpt_turbo")

templateOne = """Scenario:
You are a Vanguard helper clerk assisting a user with their Vanguard questions.
Vanguard is an investment firm.
Instructions:
Convert the clients message into a search key in relation to actions the client can consider with the website. You have to key in the users intent into a vector database to get the results you want
Provide your response in the form of words separated by commas, produce up to 8 results.
Customer Message:
{text}
Question:
What actions would you suggest to the customer based on their message?"""

promptOne = PromptTemplate(input_variables=["text"], template=templateOne)
chainOne = LLMChain(llm=gpt_turbo, verbose=False,prompt= promptOne)

templateTwo = """You are a customer support specialist chatbot helping Vanguard clients with their questions while browsing the Vanguard website.
Vanguard is an investment firm.
Continue the conversation with the user by answering their question.
Use the following pieces of context and recommended actions and terms to answer the user's question if necessary otherwise ignore them.
If you don't know the answer, just say “Sorry, con’t answer your question, try to ask it in a different way”, don't try to make up an answer.
context: 
{context}
Recommendation:
{searchTermsResult} 
conversation history:
{chat_history} 
Human: {question}
Vanguard Chatbot:"""

promptTwo = PromptTemplate(
    input_variables=["chat_history", "question", "context", "searchTermsResult"],
    template=templateTwo
)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
chainTwo = load_qa_chain(gpt_turbo, chain_type="stuff", memory=memory, verbose=False, prompt=promptTwo)

def chat(query):
    
    searchTerms = "".join(chainOne.run(query))
    st = searchTerms.split(",")
    matching_docs = []

    for s in st:
        seen_content = set()
        unique_documents = []
        documents = db.similarity_search(s, k=2)
        for doc in documents:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_documents.append(doc)

        matching_docs += unique_documents



    response = chainTwo(
        {
            "input_documents": matching_docs,
            "question": query,
            "searchTermsResult": searchTerms,
        },
        return_only_outputs=True,
    )
    
    return response['output_text']


def handler(event, context):
    chat_response = chat(event['queryStringParameters']["query"])
    chat_response_json = json.dumps(chat_response)
    
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": chat_response_json,
    }

#print(handler(None,None))
