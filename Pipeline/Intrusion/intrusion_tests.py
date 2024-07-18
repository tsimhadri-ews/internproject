def test_preprocess(): 
    from pandas.api.types import is_numeric_dtype
    import requests
    import sys
    import unittest
    
    class TestIntrusionDetection(unittest.TestCase):
        url = "https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/intrusiondetection.py"
        response = requests.get(url)
        lines = response.text.strip().split("\n")
        packages = ""
        functions = {}
        function_name = ""
        inside_function = False

        for line in lines:
            if line.strip().startswith("import ") or line.strip().startswith("from ") :
                packages += line + "\n"
            if line.strip().startswith("def "):
                function_name = line.replace('(', ' ').split()[1]
                functions[function_name] = packages + line + "\n"
                inside_function = True
            elif inside_function and line.strip().startswith("def "):
                functions[function_name] = functions.get(function_name) + line
                function_name = ""
                inside_function = False
            elif inside_function and not line.strip().startswith("\"\"\""):
                functions[function_name] = functions.get(function_name) + line + "\n"

        exec(functions.get("column_names"), globals())
        exec(functions.get("zscore_normalization"), globals())
        exec(functions.get("encode_text"), globals())
        exec(functions.get("preprocess"), globals())
            
        def test_df_shape(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            self.assertEqual(df.shape, (494021, 42))    
        
        def test_value_counts(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            self.assertEqual(df['outcome'].value_counts()[0], 97278)
            self.assertEqual(df['outcome'].value_counts()[1], 396743)
        
        def test_normalization(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df_normal = preprocess(df.copy())
            X = df_normal.drop(columns=["outcome"])
            for label, content in X.describe().items():
                if is_numeric_dtype(df[label]):
                    self.assertAlmostEqual(content['mean'], 0)
                    self.assertAlmostEqual(content['std'], 1)
                else:
                    self.assertEqual(content['min'], 0)
        
        def test_components_drop(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            self.assertEqual(df.shape, (494021, 23))

    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntrusionDetection)
    unittest.TextTestRunner(verbosity=4,stream=sys.stderr).run(suite)

def test_gbc():
    import requests
    import sys
    import unittest
    
    class TestIntrusionDetection(unittest.TestCase):
        url = "https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/intrusiondetection.py"
        response = requests.get(url)
        lines = response.text.strip().split("\n")
        packages = ""
        functions = {}
        function_name = ""
        inside_function = False

        for line in lines:
            if line.strip().startswith("import ") or line.strip().startswith("from ") :
                packages += line + "\n"
            if line.strip().startswith("def "):
                function_name = line.replace('(', ' ').split()[1]
                functions[function_name] = packages + line + "\n"
                inside_function = True
            elif inside_function and line.strip().startswith("def "):
                functions[function_name] = functions.get(function_name) + line
                function_name = ""
                inside_function = False
            elif inside_function and not line.strip().startswith("\"\"\""):
                functions[function_name] = functions.get(function_name) + line + "\n"

        exec(functions.get("column_names"), globals())
        exec(functions.get("zscore_normalization"), globals())
        exec(functions.get("encode_text"), globals())
        exec(functions.get("preprocess"), globals())
        exec(functions.get("traintest_split"), globals())
        
        def test_gbc_training(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            X_train, X_test, y_train, y_test = traintest_split(df)
            model = GradientBoostingClassifier(random_state=42)
            model.fit(X_train, y_train)
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, 'feature_importances_'))
        
        def test_gbc_prediction(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            X_train, X_test, y_train, y_test = traintest_split(df)
            model = GradientBoostingClassifier(random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            self.assertEqual(len(predictions), len(X_test))
        
        def test_gbc_accuracy(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            X_train, X_test, y_train, y_test = traintest_split(df)
            model = GradientBoostingClassifier(random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            self.assertGreater(accuracy, 0.9)
            f1 = f1_score(y_test, predictions)
            self.assertGreaterEqual(accuracy, 0.9)
            self.assertGreaterEqual(f1, 0.9)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntrusionDetection)
    unittest.TextTestRunner(verbosity=4,stream=sys.stderr).run(suite)

def test_nb():
    import requests
    import sys
    import unittest
    
    class TestIntrusionDetection(unittest.TestCase):
        url = "https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/intrusiondetection.py"
        response = requests.get(url)
        lines = response.text.strip().split("\n")
        packages = ""
        functions = {}
        function_name = ""
        inside_function = False

        for line in lines:
            if line.strip().startswith("import ") or line.strip().startswith("from ") :
                packages += line + "\n"
            if line.strip().startswith("def "):
                function_name = line.replace('(', ' ').split()[1]
                functions[function_name] = packages + line + "\n"
                inside_function = True
            elif inside_function and line.strip().startswith("def "):
                functions[function_name] = functions.get(function_name) + line
                function_name = ""
                inside_function = False
            elif inside_function and not line.strip().startswith("\"\"\""):
                functions[function_name] = functions.get(function_name) + line + "\n"

        exec(functions.get("column_names"), globals())
        exec(functions.get("zscore_normalization"), globals())
        exec(functions.get("encode_text"), globals())
        exec(functions.get("preprocess"), globals())
        exec(functions.get("traintest_split"), globals())
        
        def test_nb_training(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            X_train, X_test, y_train, y_test = traintest_split(df)
            model = GaussianNB()
            model.fit(X_train, y_train)
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, 'class_prior_'))
        
        def test_nb_prediction(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            X_train, X_test, y_train, y_test = traintest_split(df)
            model = GaussianNB()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            self.assertEqual(len(predictions), len(X_test))
        
        def test_nb_accuracy(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            X_train, X_test, y_train, y_test = traintest_split(df)
            model = GaussianNB()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            self.assertGreaterEqual(accuracy, 0.9)
            self.assertGreaterEqual(f1, 0.9)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntrusionDetection)
    unittest.TextTestRunner(verbosity=4,stream=sys.stderr).run(suite)

def test_ann():
    from keras import callbacks
    import requests
    import sys
    import unittest
    
    class TestIntrusionDetection(unittest.TestCase):
        url = "https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/intrusiondetection.py"
        response = requests.get(url)
        lines = response.text.strip().split("\n")
        packages = ""
        functions = {}
        function_name = ""
        inside_function = False

        for line in lines:
            if line.strip().startswith("import ") or line.strip().startswith("from ") :
                packages += line + "\n"
            if line.strip().startswith("def "):
                function_name = line.replace('(', ' ').split()[1]
                functions[function_name] = packages + line + "\n"
                inside_function = True
            elif inside_function and line.strip().startswith("def "):
                functions[function_name] = functions.get(function_name) + line
                function_name = ""
                inside_function = False
            elif inside_function and not line.strip().startswith("\"\"\""):
                functions[function_name] = functions.get(function_name) + line + "\n"

        exec(functions.get("column_names"), globals())
        exec(functions.get("zscore_normalization"), globals())
        exec(functions.get("encode_text"), globals())
        exec(functions.get("preprocess"), globals())
        exec(functions.get("traintest_split"), globals())
        
        def test_ann_training(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            X_train, X_test, y_train, y_test = traintest_split(df)
            input_shape = [X_train.shape[1]]
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=64, activation='relu', input_shape=input_shape),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=1)
            ])
            model.build()
            model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])  
            earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                                        mode="min",
                                                        patience=5,
                                                        restore_best_weights=True)
            history = model.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=256, epochs=60,callbacks=[earlystopping], verbose=0)
            self.assertIsNotNone(model)
        
        def test_ann_prediction(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            X_train, X_test, y_train, y_test = traintest_split(df)
            input_shape = [X_train.shape[1]]
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=64, activation='relu', input_shape=input_shape),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=1)
            ])
            model.build()
            model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])  
            earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                                        mode="min",
                                                        patience=5,
                                                        restore_best_weights=True)
            history = model.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=256, epochs=60,callbacks=[earlystopping], verbose=0)
            predictions = (model.predict(X_test) > 0.5).astype("int32")
            self.assertEqual(len(predictions), len(X_test))
        
        def test_ann_accuracy(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            X_train, X_test, y_train, y_test = traintest_split(df)
            input_shape = [X_train.shape[1]]
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=64, activation='relu', input_shape=input_shape),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=1)
            ])
            model.build()
            model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])  
            earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                                        mode="min",
                                                        patience=5,
                                                        restore_best_weights=True)
            history = model.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=256, epochs=60,callbacks=[earlystopping], verbose=0)
            self.assertGreater(history.history['accuracy'][-1], 0.9)
        
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntrusionDetection)
    unittest.TextTestRunner(verbosity=4,stream=sys.stderr).run(suite)

def test_svm():
    import requests
    import sys
    import unittest
    
    class TestIntrusionDetection(unittest.TestCase):
        url = "https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/intrusiondetection.py"
        response = requests.get(url)
        lines = response.text.strip().split("\n")
        packages = ""
        functions = {}
        function_name = ""
        inside_function = False

        for line in lines:
            if line.strip().startswith("import ") or line.strip().startswith("from ") :
                packages += line + "\n"
            if line.strip().startswith("def "):
                function_name = line.replace('(', ' ').split()[1]
                functions[function_name] = packages + line + "\n"
                inside_function = True
            elif inside_function and line.strip().startswith("def "):
                functions[function_name] = functions.get(function_name) + line
                function_name = ""
                inside_function = False
            elif inside_function and not line.strip().startswith("\"\"\""):
                functions[function_name] = functions.get(function_name) + line + "\n"

        exec(functions.get("column_names"), globals())
        exec(functions.get("zscore_normalization"), globals())
        exec(functions.get("encode_text"), globals())
        exec(functions.get("preprocess"), globals())
        exec(functions.get("traintest_split"), globals())
        
        def test_svm_training(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            df = df.sample(100000)
            X_train, X_test, y_train, y_test = traintest_split(df)
            model = SVC(random_state=42)
            model.fit(X_train, y_train)
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, 'class_weight_'))
        
        def test_svm_prediction(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            df = df.sample(100000)
            X_train, X_test, y_train, y_test = traintest_split(df)
            model = SVC(random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            self.assertEqual(len(predictions), len(X_test))
    
        def test_svm_accuracy(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            df = df.sample(100000)
            X_train, X_test, y_train, y_test = traintest_split(df)
            model = SVC(kernel='linear', random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            self.assertGreaterEqual(accuracy, 0.9)
            self.assertGreaterEqual(f1, 0.9)
        
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntrusionDetection)
    unittest.TextTestRunner(verbosity=4,stream=sys.stderr).run(suite)


def test_logreg():
    import requests
    import sys
    import unittest
    
    class TestIntrusionDetection(unittest.TestCase):
        url = "https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/intrusiondetection.py"
        response = requests.get(url)
        lines = response.text.strip().split("\n")
        packages = ""
        functions = {}
        function_name = ""
        inside_function = False

        for line in lines:
            if line.strip().startswith("import ") or line.strip().startswith("from ") :
                packages += line + "\n"
            if line.strip().startswith("def "):
                function_name = line.replace('(', ' ').split()[1]
                functions[function_name] = packages + line + "\n"
                inside_function = True
            elif inside_function and line.strip().startswith("def "):
                functions[function_name] = functions.get(function_name) + line
                function_name = ""
                inside_function = False
            elif inside_function and not line.strip().startswith("\"\"\""):
                functions[function_name] = functions.get(function_name) + line + "\n"

        exec(functions.get("column_names"), globals())
        exec(functions.get("zscore_normalization"), globals())
        exec(functions.get("encode_text"), globals())
        exec(functions.get("preprocess"), globals())
        exec(functions.get("traintest_split"), globals())
        
        def test_logreg_training(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            X_train, X_test, y_train, y_test = traintest_split(df)
            model = LogisticRegression(random_state=42, solver='sag')
            model.fit(X_train, y_train)
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, 'intercept_'))
        
        def test_logreg_prediction(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            X_train, X_test, y_train, y_test = traintest_split(df)
            model = LogisticRegression(random_state=42, solver='sag')
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            self.assertEqual(len(predictions), len(X_test))
        
        def test_logreg_accuracy(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            X_train, X_test, y_train, y_test = traintest_split(df)
            model = LogisticRegression(random_state=42, solver='sag')
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            self.assertGreaterEqual(accuracy, 0.9)
            self.assertGreaterEqual(f1, 0.9)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntrusionDetection)
    unittest.TextTestRunner(verbosity=4,stream=sys.stderr).run(suite)

def test_rfc():
    import requests
    import sys
    import unittest
    
    class TestIntrusionDetection(unittest.TestCase):
        url = "https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/intrusiondetection.py"
        response = requests.get(url)
        lines = response.text.strip().split("\n")
        packages = ""
        functions = {}
        function_name = ""
        inside_function = False

        for line in lines:
            if line.strip().startswith("import ") or line.strip().startswith("from ") :
                packages += line + "\n"
            if line.strip().startswith("def "):
                function_name = line.replace('(', ' ').split()[1]
                functions[function_name] = packages + line + "\n"
                inside_function = True
            elif inside_function and line.strip().startswith("def "):
                functions[function_name] = functions.get(function_name) + line
                function_name = ""
                inside_function = False
            elif inside_function and not line.strip().startswith("\"\"\""):
                functions[function_name] = functions.get(function_name) + line + "\n"

        exec(functions.get("column_names"), globals())
        exec(functions.get("zscore_normalization"), globals())
        exec(functions.get("encode_text"), globals())
        exec(functions.get("preprocess"), globals())
        exec(functions.get("traintest_split"), globals())
        
        def test_rfc_training(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            X_train, X_test, y_train, y_test = traintest_split(df)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, 'max_depth'))
        
        def test_rfc_prediction(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            X_train, X_test, y_train, y_test = traintest_split(df)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            self.assertEqual(len(predictions), len(X_test))
        
        def test_rfc_accuracy(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            X_train, X_test, y_train, y_test = traintest_split(df)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            self.assertGreaterEqual(accuracy, 0.9)
            self.assertGreaterEqual(f1, 0.9)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntrusionDetection)
    unittest.TextTestRunner(verbosity=4,stream=sys.stderr).run(suite)

def test_dtc():
    import requests
    import sys
    import unittest
    
    class TestIntrusionDetection(unittest.TestCase):
        url = "https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/intrusiondetection.py"
        response = requests.get(url)
        lines = response.text.strip().split("\n")
        packages = ""
        functions = {}
        function_name = ""
        inside_function = False

        for line in lines:
            if line.strip().startswith("import ") or line.strip().startswith("from ") :
                packages += line + "\n"
            if line.strip().startswith("def "):
                function_name = line.replace('(', ' ').split()[1]
                functions[function_name] = packages + line + "\n"
                inside_function = True
            elif inside_function and line.strip().startswith("def "):
                functions[function_name] = functions.get(function_name) + line
                function_name = ""
                inside_function = False
            elif inside_function and not line.strip().startswith("\"\"\""):
                functions[function_name] = functions.get(function_name) + line + "\n"

        exec(functions.get("column_names"), globals())
        exec(functions.get("zscore_normalization"), globals())
        exec(functions.get("encode_text"), globals())
        exec(functions.get("preprocess"), globals())
        exec(functions.get("traintest_split"), globals())
        
        def test_dtc_training(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            X_train, X_test, y_train, y_test = traintest_split(df)
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, y_train)
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, 'max_depth'))
        
        def test_dtc_prediction(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            X_train, X_test, y_train, y_test = traintest_split(df)
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            self.assertEqual(len(predictions), len(X_test))
        
        def test_dtc_accuracy(self):
            cols = column_names()
            file_path = 'https://raw.githubusercontent.com/tsimhadri-ews/internproject/intrusion-detection-0/src/kddcup.data_10_percent_corrected.csv'
            df = pd.read_csv(file_path, names=cols)
            df = preprocess(df)
            X_train, X_test, y_train, y_test = traintest_split(df)
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            self.assertGreater(accuracy, 0.9)
            f1 = f1_score(y_test, predictions)
            self.assertGreaterEqual(accuracy, 0.9)
            self.assertGreaterEqual(f1, 0.9)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntrusionDetection)
    unittest.TextTestRunner(verbosity=4,stream=sys.stderr).run(suite)
