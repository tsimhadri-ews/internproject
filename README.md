# End-to-end MLOps Pipeline for Cybersecurity Classification

## Project Description
This project aims to classify (binary) several cybersecurity-adjacent datasets. It leverages a robust MLOps Level 2 end-to-end pipeline, ensuring automated retraining, CI/CD, and efficient model deployment. The datasets used for classification include:
- **KDDCup1999**: Intrusion detection
- **Virusshare.com**: Malware detection
- **Mendeley data**: Web page phishing detection
- **UNSWNB-15**: Multi-classification of network intrusions

## Installation Instructions
To set up this project, follow these steps:

1. **Setup EC2 Resources and Anaconda**:
   - Follow the documentation to set up anaconda and create an EC2 instance and install a Ubuntu virtual machine on it.
1. **Install Kubeflow**:
   - Follow the documentation to prepare your EC2 instance and install KubeFlow.
2. **Install Kafka**:
   - Follow the documentation to install Kafka on the KubeFlow instance.
3. **Create Kafka Brokers**:
   - Follow the documentation to create three brokers on the Kafka cluster.
4. **Create Kafka Topic**:
   - Follow the documentation to create a topic for each desired use case with a replication factor of 3.
5. **Configure S3 Bucket Permissions**:
   - Follow the documentation to configure s3 permissions.
6. **Create PostgreSQL Database**:
   - Follow the documentation to create the database and install pgAdmin on the cluster.
7. **Configure AWS Secrets**:
   - Follow the documentation to allow credentials to be securely read in.

## Usage
To use this project, follow these steps:

1. Ensure all necessary installations are completed.
2. Run the python files for the desired uses in CreateDatabases to populate tables in PostgreSQL.
3. Run the jupyter notebooks for the desired use cases in the experiment and folder to create pipeline components and trigger a run.
4. Run the HITL python file for the desired use cases to create the HITL for prediction confirmation and triggering.
5. Edit the python files and the jupter notebooks in the Inference folder for the desired use cases to update the unseen data sources and run for inferencing.
6. Monitor the pipeline for automated retraining and CI/CD processes.

### Example Usage
1. **Data Ingestion**:
   - Use Kafka for streaming data into the pipeline.
   - Example: `KafkaProducer` to send data to the pipeline.
2. **Data Processing**:
   - Utilize pandas and numpy for data manipulation and preprocessing, and push to git to trigger git actions.
   - Example: Apply `boxcox` transformation to normalize data.s
4. **Database Integration**:
   - Store and retrieve data using PostgreSQL.
   - Example: Use `sqlalchemy` to interact with the PostgreSQL database.

## Features
- Binary classification of various cybersecurity datasets.
- End-to-end MLOps Level 2 functionality.
- Automated retraining of models.
- Continuous Integration/Continuous Deployment (CI/CD).
- Scalable and efficient data ingestion and processing.
- Real-time data streaming and batch processing.

## Technologies Used
The following technologies and libraries are used in this project:
```python
import logging
import requests
import pandas as pd
from kafka import KafkaProducer
import json
import psycopg2
from sqlalchemy import create_engine, text
from scipy.special import boxcox
import numpy as np
import uuid
from io import BytesIO
from zipfile import ZipFile
```
These can be installed from our requirements.txt file.
We recommend creating a python virtual environment.
```
python -m venv [pipeline]
source pipeline/bin/activate
```
And then run the install. 
```
pip install -r requirements.txt
```




## Contributing
We welcome contributions to this project! To contribute, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push your branch to your fork.
4. Submit a pull request with a detailed description of your changes.

## License
This project is licensed under the Kubeflow license. For more details, refer to the LICENSE file in the repository.

## Authors and Acknowledgments
This project was developed by the intern team at Everwatch Corporation:

- Myra Cropper
- Jonathan Mei
- Sachin Ashok
- Jonathan Rogers
- Tanvi Simhadri
- Kyle Simon
- Connor Mullikin
- Matthew Lessler
- Izaiah Davis
- Quinn Dunnigan

Mentored by David Culver.

## Contact Information
For questions or issues, please open an issue in the repository or contact the authors.
