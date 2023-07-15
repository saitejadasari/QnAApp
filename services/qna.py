from transformers import pipeline
import os

MODEL_1 = 'bert-large-uncased-whole-word-masking-finetuned-squad'


class QnA:
    def __init__(self, model=MODEL_1):
        self.qa = pipeline('question-answering', model=model)

    def answer_question(self, question, context):
        response = self.qa(context=context, question=question)
        return response
