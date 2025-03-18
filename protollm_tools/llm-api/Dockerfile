FROM nginx/unit:1.28.0-python3.10

COPY protollm_api /app/protollm_api
COPY requirements.txt /app
COPY unit_config.json /docker-entrypoint.d/config.json
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt install git

COPY unit_config.json /docker-entrypoint.d/config.json