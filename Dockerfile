# Using the Ubuntu image (our OS)
FROM ubuntu:20.04
# Update package manager (apt-get) 
# and install (with the yes flag `-y`)
# Python and Pip
RUN apt-get update && apt-get install -y \
    python3.7 \
    python3-pip

# =====
# The new stuff is below
# =====

# Install our Python dependencies
#RUN pip install Requests Pygments
RUN pip install kafka-python
# Copy our script into the container
COPY kakfa_provaConsumer.py /kakfa_provaConsumer.py

# Run the script when the image is run
ENTRYPOINT ["python3", "/kakfa_provaConsumer.py"]