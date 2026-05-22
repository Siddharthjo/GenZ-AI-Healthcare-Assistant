import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import app as flask_app


@pytest.fixture(scope='session')
def client():
    flask_app.app.config['TESTING'] = True
    flask_app.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with flask_app.app.test_client() as c:
        with flask_app.app.app_context():
            flask_app.db.create_all()
        yield c
