FROM python:3.10

COPY fastapi_app.py my_requests.py online_inference/
COPY requirements.txt online_inference/
COPY models/model.pkl online_inference/models/model.pkl
COPY input_data/heart_cleveland_upload.csv online_inference/input_data/heart_cleveland_upload.csv
WORKDIR online_inference/

ENV MODEL_PATH="models/model.pkl"
ENV DATA_PATH="input_data/heart_cleveland_upload.csv"

RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "fastapi_app:ml_project_app", "--host", "0.0.0.0", "--port", "8000"]