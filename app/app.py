from flask import Flask
from api.api import API

def create_app():
    app = Flask(__name__)
    API(app)
    return app
