from flask import Flask, request, render_template

from controller import textController
from services import qna as qna

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.post('/context/answer')
def answer_question():
    req = request.json
    ques = req['question']
    context = req['context']
    print(context)
    print(ques)
    result = textController.answer_question(ques, context)
    print(result,"res")
    return result


# /upload text and store
@app.post('/text/upload')
def text_upload():
    req = request.json
    # add validation
    response = textController.upload_text(req)
    return response


# /answer the questions from the file
@app.post('/text/answer')
def text_answer():
    req = request.json
    # add validation
    question = req['question']
    index = req['index_name']
    response = textController.answer_question_text(question, index)
    return response


@app.get('/index/status')
def get_index_status():
    response = textController.get_index_status()
    return str(response)


if __name__ == '__main__':
    app.run(debug=True)
    app.config.from_pyfile('app.conf')
