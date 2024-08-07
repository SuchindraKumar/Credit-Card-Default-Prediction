import sys
from flask import Flask, request, render_template
from flask_pymongo import PyMongo
from pymongo.mongo_client import MongoClient
from src.pipeline.prediction_pipeline import CustomDataset, PredictPipeline
from src.exception import CustomException
from src.logger import logging
from dotenv import load_dotenv
import os

application = Flask(__name__)

load_dotenv()

app = application

# MongoDB connection
MONGO_URI = os.environ.get('MONGO_URI')
client = MongoClient(MONGO_URI)
db = client['credit_card_default']
app.config['MONGO_URI'] = MONGO_URI
mongo = PyMongo(app)

# Define global error handler
@app.errorhandler(Exception)
def handle_error(e):
    logging.error("An error occurred: %s", str(e))
    return "An error occurred", 500

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/form', methods=['GET'])
def predict_datapoint():
    return render_template('form.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    try:
        data = CustomDataset()
        data_frame = data.get_data_as_dataframe(**request.form)
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(data_frame)

        result = prediction.tolist()
        if result[0] == 1:
            prediction_result = "This Individual Won't Pay the Credit Bill Next Month"
        else:
            prediction_result = "This Individual Will Pay the Credit Bill Next Month"

        # Save data to MongoDB
        entry = request.form.to_dict()
        entry['prediction_result'] = prediction_result
        logging.info("Saving data to MongoDB...")
        db.credit_card_data.insert_one(entry)
        logging.info("Data saved to MongoDB successfully.")

        logging.info("Data saved to MongoDB")

        return render_template('result.html', final_result=prediction_result)

    except Exception as e:
        logging.error("An error occurred during prediction: %s", str(e))
        raise CustomException(e, sys)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080, debug=True)


# Author : Suchindra Kumar  
