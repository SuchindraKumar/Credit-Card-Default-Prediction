import sys
from flask import Flask, request, render_template
from flask_pymongo import PyMongo
from src.pipeline.prediction_pipeline import CustomDataset, PredictPipeline
from src.exception import CustomException
from src.logger import logging

application = Flask(__name__)
app = application

# MongoDB connection
app.config['MONGO_URI'] = 'mongodb://localhost:27017/credit_card_db'
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
        if result[0] > 0.5:
            prediction_result = "This Individual Won't Pay the Credit Bill Next Month"
        else:
            prediction_result = "This Individual Will Pay the Credit Bill Next Month"

        # Save data to MongoDB
        entry = request.form.to_dict()
        entry['prediction_result'] = prediction_result
        mongo.db.credit_card_data.insert_one(entry)

        logging.info("Data saved to MongoDB")

        return render_template('result.html', final_result=prediction_result)

    except Exception as e:
        logging.error("An error occurred during prediction: %s", str(e))
        raise CustomException(e, sys)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
