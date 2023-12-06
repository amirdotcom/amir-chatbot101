from flask import Flask, render_template, request, jsonify
from chat import get_response

app = Flask(__name__)

# define route
@app.get("/") #get request
def index_get():
    return render_template("base.html")  #return base.html template

# route to do the prediction
@app.post("/predict") #post request
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    response = get_response(text)
    messsage = {"answer": response}
    return jsonify(messsage)

if __name__=="__main__":
    app.run(debug=True) #to run the development

