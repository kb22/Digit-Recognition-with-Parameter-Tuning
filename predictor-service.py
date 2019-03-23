from flask import Flask, request
from flask_restplus import Api, Resource, fields
from sklearn.externals import joblib
import numpy as np

flask_app = Flask(__name__)
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "Digit Recognizer", 
		  description = "A tool that recognizes numbers 0 - 9 from an image")

name_space = app.namespace('prediction', description='Make predictions')

image_data = app.model('Image data', 
				  {'data': fields.List(fields.Integer(description = "Pixel value"),
	  								   required = True, 
    					  			   description = "Image pixel values")})

prediction_model = joblib.load("digit-recognizer.joblib")

@name_space.route("/predict")
class MainClass(Resource):

	@app.doc(responses={ 200: 'OK', 400: 'Error' })
	@app.expect(image_data)
	def post(self):
		try:
			image_data = request.json['data']
			prediction = prediction_model.predict(np.array(image_data).reshape(1, -1))
			return {
				"prediction": str(prediction[0])
			}
		except Exception as e:
			name_space.abort(400, 
							 message = "Make sure image size is 28x28", 
							 status = "Could not make prediction", 
							 statusCode = "400")