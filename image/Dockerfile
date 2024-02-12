FROM public.ecr.aws/lambda/python:3.11

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN yum install gcc-c++ -y
# Install the specified packages
RUN pip install -r requirements.txt

# Copy all files and folders in ./src
COPY src ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler.
CMD [ "main.handler" ]
