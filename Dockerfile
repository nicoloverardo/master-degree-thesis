FROM heroku/miniconda
FROM python:3.8

# Grab requirements.txt.
ADD requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install -r /tmp/requirements.txt

# Add our code
ADD . .
WORKDIR .

CMD streamlit run app.py
