from services.qna import QnA


# extracts answer from the context passage
def extract_answer(question, context, sort=True):
    results = []
    qna = QnA()
    for c in context:
        # feed the reader the question and contexts to extract answers
        answer = qna.answer_question(question=question, context=c)
        # add the context to answer dict for printing both together
        answer["context"] = c
        results.append(answer)
    if sort:
        # sort the result based on the score from reader model
        results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results



