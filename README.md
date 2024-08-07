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
   - Follow the documentation to set up [Anaconda](https://everwatchsolutions-my.sharepoint.com/personal/kyle_simon_everwatchsolutions_com/_layouts/15/Doc.aspx?sourcedoc={d4b5b174-d832-4649-bd8b-4775bb688c20}&action=edit&wd=target%28AWS%20EC2%20Setup.one%7C4ff51a2b-f919-42fd-8445-e3297aa05de4%2FInstalling%20Anaconda%7C7f1a4c5e-a3ce-499d-a034-1f4737c0200b%2F%29&wdorigin=NavigationUrl) and [create an EC2 instance](https://github.com/tsimhadri-ews/internproject/tree/main/Documentation/EC2_and_Anaconda) and install a Ubuntu virtual machine on it.
2. **Install Kubeflow**:
   - Follow the [documentation](https://github.com/tsimhadri-ews/internproject/blob/main/Documentation/Kubeflow/Kubernetes%20and%20Kubeflow%20-%20Setup.pdf) to prepare your EC2 instance and install KubeFlow.
3. **Install Kafka**:
   - Install [Kafka](https://github.com/tsimhadri-ews/internproject/blob/main/Documentation/Kafka/Kafka%20Workflow_%20Installing%20Kafka%20on%20Ubuntu%2022.pdf) on the KubeFlow instance.
3. **Create Kafka Brokers**:
   - Follow the documentation to [create three brokers](https://github.com/tsimhadri-ews/internproject/tree/main/Documentation/Kafka) on the Kafka cluster.
4. **Create Kafka Topic**:
   - Follow the documentation to [create a topic]https://github.com/tsimhadri-ews/internproject/tree/main/Documentation/Kafkal) for each desired use case with a replication factor of 3.
5. **Configure S3 Bucket Permissions**:
   - Follow the documentation to [configure s3 permissions](https://everwatchsolutions-my.sharepoint.com/personal/kyle_simon_everwatchsolutions_com/_layouts/15/Doc.aspx?sourcedoc={d4b5b174-d832-4649-bd8b-4775bb688c20}&action=edit&wd=target%28Databases.one%7C5ea6a88c-1db6-4e3b-8d85-6a0ce6be28e0%2FS3%20Buckets%20permissions%7Ca79bc9d2-fb3b-4081-a741-4c143ef8a2db%2F%29&wdorigin=NavigationUrl).
6. **Create PostgreSQL Database**:
   - Follow the [documentation](https://github.com/tsimhadri-ews/internproject/blob/main/Documentation/Database/Database%20-%20Setting%20up%20a%20Database.pdf) to create the database and install pgAdmin on the cluster.
7. **Configure AWS Secrets**:
   - Follow the [documentation](https://everwatchsolutions-my.sharepoint.com/personal/kyle_simon_everwatchsolutions_com/_layouts/15/Doc.aspx?sourcedoc={d4b5b174-d832-4649-bd8b-4775bb688c20}&action=edit&wd=target%28Databases.one%7C5ea6a88c-1db6-4e3b-8d85-6a0ce6be28e0%2FRead%20in%20Credentials%20securely%7Cc84ae31e-2c25-4ca6-b6df-442336f42020%2F%29&wdorigin=NavigationUrl) to allow credentials to be securely read in.

## Usage
To use this project, follow these steps:

1. Ensure all necessary installations are completed.
2. Create an S3 bucket in accordance to the naming convention for the specific use case.
3. Run the python files for the desired uses in CreateDatabases to populate tables in PostgreSQL.
4. Run the jupyter notebooks for the desired use cases in the pipelines folder to create pipeline components and trigger a run.
5. Run the HITL python file for the desired use cases to create the HITL for prediction confirmation and triggering.
6. Edit the python files and the jupter notebooks in the Inference folder for the desired use cases to update the unseen data sources and run for inferencing.
7. Monitor the pipeline for automated retraining and CI/CD processes.

## Example Usage

### Data Ingestion:

#### Use Kafka for streaming data into the pipeline.

```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))
data = {"example_key": "example_value"}
producer.send('your_topic', value=data)
```

### Data Processing:
#### Utilize pandas and numpy for data manipulation and preprocessing, and push to Git to trigger Git actions.
```python
import pandas as pd
import numpy as np
from scipy.special import boxcox

df = pd.read_csv('data.csv')
df['normalized_feature'] = boxcox(df['feature_column'] + 1)[0]

# Commit and push changes to trigger CI/CD
!git add data.csv
!git commit -m "Processed data"
!git push
```
### Database Integration:
#### Store and retrieve data using PostgreSQL.
```python
from sqlalchemy import create_engine, text

engine = create_engine('postgresql+psycopg2://username:password@localhost:5432/database_name')
with engine.connect() as connection:
    result = connection.execute(text("SELECT * FROM table_name"))
    for row in result:
        print(row)
```
### Pipeline Execution:
#### Push changes to the Jupyter notebook in the pipelines folder to create pipeline components and trigger a run.
```python
# Edit the notebook and save changes to trigger pipeline execution
!git add pipelines/your_notebook.ipynb
!git commit -m "Updated pipeline notebook"
!git push
```
### Human-in-the-Loop (HITL) for Prediction Confirmation:
#### Run the HITL python file for the desired use cases to create the HITL for prediction confirmation and triggering.
```python
# Example HITL script
import json

def hitl_confirmation(prediction):
    # Logic for human confirmation
    confirmed = input(f"Confirm prediction {prediction}: (yes/no) ")
    return confirmed.lower() == 'yes'

with open('predictions.json', 'r') as f:
    predictions = json.load(f)

confirmed_predictions = [pred for pred in predictions if hitl_confirmation(pred)]

with open('confirmed_predictions.json', 'w') as f:
    json.dump(confirmed_predictions, f)
```

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
