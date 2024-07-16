from flask import Flask,render_template,request
import pickle
import numpy as np
model=pickle.load(open("traning.pkl","rb"))
app=Flask(__name__)
@app.route('/',methods=['GET'])
def home():
    return render_template("home.html")
@app.route("/predict",methods=["POST"])
def predict():
    features=[x for x in request.form.values()]
    fea=[np.array(features)]
    preds=model.predict(fea)
    if preds==1:
        predics="are likely to"
    else:
        predics="likely will not"

    return render_template('home.html',prediction=predics)
if __name__=="__main__":
    app.run(debug=True)