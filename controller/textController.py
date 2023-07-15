from utils import util
from services.pineconeService import PineconeService
from services import answerService


def upload_text(request):
    response = {'status': 200}
    try:
        context = request['context']
        # index_name = util.generate_uuid() # only 1 is allowed in free tier
        retriever = util.get_retriever()
        pinecone = PineconeService()
        # create index name if new index
        index, index_name = pinecone.get_index(retriever)
        chunks = util.split_text(context)
        print("chunks", len(chunks))
        pinecone.embed_and_store(index, retriever, chunks)
        response['index_name'] = index_name
    except Exception as e:
        print("Exception in upload_text", e)
        response['status'] = 500
    return response


def answer_question_text(question, index_name):
    retriever = util.get_retriever()
    pinecone = PineconeService()
    index, _ = pinecone.get_index(retriever, index_name=index_name)
    context = pinecone.get_context(index, retriever, question)
    print("context", context)
    response = answerService.extract_answer(question, context)
    return response


def answer_question(question, context):
    response = answerService.extract_answer(question, [context], sort=False)
    return response


def get_index_status():
    pinecone = PineconeService()
    index, index_name = pinecone.get_index(None)
    return pinecone.get_index_status(index)
