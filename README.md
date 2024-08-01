# End-to-end MLOps Pipeline for Cybersecurity Classification

## Project Description
This project aims to classify (binary) several cybersecurity-adjacent datasets. It leverages a robust MLOps Level 2 end-to-end pipeline, ensuring automated retraining, CI/CD, and efficient model deployment. The datasets used for classification include:
- **KDDCup1999**: Intrusion detection
- **Virusshare.com**: Malware detection
- **Mendeley data**: Web page phishing detection
- **UNSWNB-15**: Multi-classification of network intrusions

## Installation Instructions
To set up this project, follow these steps:

1. **Install Kubeflow**:
   - Follow the official [Kubeflow installation guide](https://www.kubeflow.org/docs/started/getting-started/).
2. **Install Kafka**:
   - Refer to the [Kafka quickstart guide](https://kafka.apache.org/quickstart).
3. **Install KServe**:
   - Check out the [KServe documentation](https://kserve.github.io/website/).
4. **Install PostgreSQL**:
   - Use the [PostgreSQL installation instructions](https://www.postgresql.org/download/).

## Usage
To use this project, follow these steps:

1. Ensure all necessary installations are completed.
2. Configure the MLOps pipeline with the installed components.
3. Execute the classification tasks on the specified datasets.
4. Monitor the pipeline for automated retraining and CI/CD processes.

### Example Usage
1. **Data Ingestion**:
   - Use Kafka for streaming data into the pipeline.
   - Example: `KafkaProducer` to send data to the pipeline.
2. **Data Processing**:
   - Utilize pandas and numpy for data manipulation and preprocessing.
   - Example: Apply `boxcox` transformation to normalize data.
3. **Model Training and Deployment**:
   - Use KServe for model serving.
   - Example: Deploy trained models using KServe.
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
