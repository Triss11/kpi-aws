FROM sohini11/test-images:1.0.11
RUN rm -rf /usr/src/main/ usr/src/main/kpi-aws

RUN git clone https://github.com/Triss11/kpi-aws.git /usr/src/main/kpi-aws
#COPY layoutlm.pt /usr/src/main/kpi-aws/layoutlm.pt

WORKDIR /usr/src/main/kpi-aws

#COPY requirements/requirements.txt /usr/src/main/requirements.txt

#RUN pip install -no-cache-dir-upgrade pip && A pip install -r requirements.txt
RUN apt-get update &&\
apt-get install -y tesseract-ocr && pip install prometheus-client

EXPOSE 5000 8080

CMD ["gunicorn", "--bind", "0.0.0.0:5000","main:app"]
