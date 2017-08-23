# QASystem
## Upgraded Version of QASystem3
1. TensorFlow 1.2
2. Python 2.7
3. Model exported for TensorFlow Serving.

### 0. Create/Modify the TensorFlow Model

This Q&A model is adapted from https://github.com/yolandawww/QASystem

### 1. Train the TensorFlow Model

This model is trained on a GPU enabled Google Cloud Compute Engine.

#### To setup a GPU enabled Google Cloud Compute Engine, please refer to this video:

https://www.youtube.com/watch?v=abEf3wQJBmE

### 2. Export the TensorFlow Model

#### To export the TensorFlow Model, please refer to these files:

https://github.com/llSourcell/How-to-Deploy-a-Tensorflow-Model-in-Production/blob/master/custom_model.py

https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/inception_saved_model.py

https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py

[qa_model.py](code/qa_model.py)  

### 3. Deploy the TensorFlow Model to Google Cloud

In order to deploy the TensorFlow Model to a Google Cloud Container Machine, a Docker container is needed to pack the exported model with all its dependencies. Docker is an open platform for developers and sysadmins to build, ship, and run distributed applications, whether on laptops, data center VMs, or the cloud.

Kubernetes is an open-source platform designed to automate deploying, scaling, and operating application containers. It is used to deploy the model contained in the Docker image to Google Cloud.

#### Please refer to these links for TensorFlow model deployment:

`How to Deploy a TensorFlow Model in Production tutorial`

https://github.com/llSourcell/How-to-Deploy-a-Tensorflow-Model-in-Production/blob/master/demo.ipynb

`Video of this tutorial`

https://www.youtube.com/watch?v=T_afaArR0E8&feature=youtu.be

`Tutorial of Serving the MNIST Model`

https://www.tensorflow.org/serving/serving_basic

`Tutorial of Serving the Inception Model`

https://www.tensorflow.org/serving/serving_inception

### 4. Request the TensorFlow Model from a Python Client File

A client file is needed to request answer prediction from the model deployed on Google Cloud remotely. This client file is in Python. You need to install `grpc.beta` and `tensorflow-serving-api` packages in order to run the client file.

#### Please refer to the following links to create a client file:

https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_client.py

https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/inception_client.py

https://github.com/liweijiang/QASystemPythonClientHeroku/blob/master/app.py

### 5. Deploy the Python Client File to Heroku

In order to connect the front end Javascript to the Python client file, we need to create a Python Flask web app and serve it on Heroku to be requested remotely.

#### Sample Flask files:

https://github.com/miguelgrinberg/flask-examples/blob/master/01-hello-world/hello.py

https://github.com/liweijiang/QASystemPythonClientHeroku/blob/master/app.py

#### How to deploy Python Flask web application to Heroku:

https://progblog.io/How-to-deploy-a-Flask-App-to-Heroku/

### 6. Request Python Client File from Javascript using JQuery Ajax Request

After the Python client is served, we can make an ajax request from Javascript to access the Python method.

```
var jqXHR = $.ajax({
    url: "https://qasystem-python-client-heroku.herokuapp.com/qatest",
    crossDomain: true,
    type: "GET",
    data: {'question': input_question},
    success: function(data) {
      response_json = JSON.parse(data);
      console.log('Do something to data.');
    },
    error: function(request, status, error) {
     console.log('Error');
     respond(messageInternalError);
    }
});
```
