import kfp
from kfp_server_api.exceptions import ApiException

try:
    client = kfp.Client(host="192.168.30.101")
    print("Client connected successfully")
    print(client)
    experiments = client.list_experiments()
    print("Experiments listed successfully")
    print(experiments)
except ApiException as e:
    print(f"Failed to connect or list experiments: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
