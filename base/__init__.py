import os

from flask import Flask

# Get the directory where this __init__.py file is located
basedir = os.path.abspath(os.path.dirname(__file__))
print(basedir)

# Create Flask app with explicit template and static folder paths
app = Flask(__name__,
            template_folder=os.path.join(basedir, 'templates'),
            static_folder=os.path.join(basedir, 'static'))

app.app_context().push()

from base import controllers
