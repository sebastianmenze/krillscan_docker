FROM python:3.8-slim 

COPY ./requirements.txt ./requirements.txt
COPY echolab2 /echolab2
COPY echopy /echopy
#COPY source_folder /source_folder
#COPY target_folder /target_folder
COPY ./settings.ini /settings.ini

COPY ./krillscan_edge_server.py /krillscan_edge_server.py

RUN pip install --upgrade -r requirements.txt

CMD ["python","krillscan_edge_server.py"]

