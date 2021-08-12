#base image
FROM python:3.9.6

#Set the working directory
WORKDIR /

#copy all the files
COPY . .

#Install the dependencies
RUN pip3 install -r requirements.txt

#Expose the required port
EXPOSE 5000

#Run the command
CMD gunicorn app:app