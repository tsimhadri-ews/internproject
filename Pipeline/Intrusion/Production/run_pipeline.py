def run_pipeline(yaml_file):
    KUBEFLOW_HOST = 'http://acc85673e1f094914a006f330bb51cb8-353421018.us-east-1.elb.amazonaws.com'
    KUBEFLOW_USERNAME = os.getenv('USER')
    KUBEFLOW_PASSWORD = os.getenv('PASSWORD')


    session = requests.Session()
    login_url = f"{KUBEFLOW_HOST}" 
    response = session.get(login_url)
    assert response.status_code == 200, f"Failed to access login page: {response.status_code}"
    print("Accessed login page")

    #apparently kubeflow doesnt want you to log onto their webpage 
    #kubeflow sucks
    soup = BeautifulSoup(response.text, 'html.parser')
    login_form = soup.find('form')
    login_action = login_form['action']
    hidden_inputs = login_form.find_all("input", type="hidden")

    login_data = {
        'login': KUBEFLOW_USERNAME,
        'password': KUBEFLOW_PASSWORD,
    }
    for hidden_input in hidden_inputs:
        login_data[hidden_input['name']] = hidden_input['value']


    login_post_url = f"{KUBEFLOW_HOST}{login_action}"
    response = session.post(login_post_url, data=login_data, allow_redirects=True)


    assert response.status_code == 200, f"Failed to log in: {response.status_code}"
    print("Logged in successfully")


    session_cookie = session.cookies.get_dict()
    print(f"Session cookie: {session_cookie}")


    if not session_cookie:
        raise ValueError("Session cookie is empty. Login failed.")

    cookie_str = '; '.join([f"{key}={value}" for key, value in session_cookie.items()])
    print(f"Formatted cookie string: {cookie_str}")

    api_endpoint = f"{KUBEFLOW_HOST}/pipeline"
    client = kfp.Client(host=api_endpoint, cookies=cookie_str)


    experiment_name = 'Test Experiment'

    try:
        experiment = client.create_experiment(name=experiment_name)
        print(f'Experiment {experiment_name} created with ID: {experiment.id}')
    except ApiException as e:
        print(f"Exception when creating experiment: {e}")
        print(f"Status: {e.status}")
        print(f"Reason: {e.reason}")
        print(f"Headers: {e.headers}")
        print(f"Body: {e.body}")
    print(client.list_experiments())

    pipeline_file = yaml_file 

    # Define the pipeline name
    pipeline_name = str(pipeline_file)

    # Upload the pipeline
    pipeline = client.upload_pipeline(pipeline_file, pipeline_name=pipeline_name)

    print(f'Pipeline {pipeline_name} uploaded successfully with ID: {pipeline.id}')
    run_name = 'Intrusion Detection Run'
    arguments = {} 

    try:
        run = client.run_pipeline(experiment.id, run_name, pipeline_file, arguments)
        print(f'Pipeline run {run_name} started with ID: {run.id}')
    except ApiException as e:
        print(f"Exception when running pipeline: {e}")
        print(f"Status: {e.status}")
        print(f"Reason: {e.reason}")
        print(f"Headers: {e.headers}")
        print(f"Body: {e.body}")