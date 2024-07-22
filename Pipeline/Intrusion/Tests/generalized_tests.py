import unittest
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf

class GeneralizedTest(unittest.TestCase):
    
    @staticmethod
    def column_names():
        # Define column names specific to your dataset
        return [
            # Add your column names here
        ]

    @staticmethod
    def zscore_normalization(df):
        # Implement z-score normalization if required
        return df

    @staticmethod
    def encode_text(df):
        # Implement text encoding if required
        return df

    @staticmethod
    def preprocess(df):
        df = GeneralizedTest.zscore_normalization(df)
        df = GeneralizedTest.encode_text(df)
        return df

    @staticmethod
    def traintest_split(df):
        # Split dataframe into features and target, then train-test split
        from sklearn.model_selection import train_test_split
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def load_data(self, file_path):
        cols = self.column_names()
        df = pd.read_csv(file_path, names=cols)
        return self.preprocess(df)

    def train_model(self, model_class, file_path, model_args={}):
        df = self.load_data(file_path)
        X_train, X_test, y_train, y_test = self.traintest_split(df)
        model = model_class(**model_args)
        model.fit(X_train, y_train)
        return model, X_test, y_test

    def test_training(self, model_class, file_path, model_args={}):
        model, X_test, y_test = self.train_model(model_class, file_path, model_args)
        self.assertIsNotNone(model)
        return model

    def test_prediction(self, model_class, file_path, model_args={}):
        model, X_test, y_test = self.train_model(model_class, file_path, model_args)
        predictions = model.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))
        return predictions

    def test_accuracy(self, model_class, file_path, model_args={}, accuracy_threshold=0.9, f1_threshold=0.9):
        model, X_test, y_test = self.train_model(model_class, file_path, model_args)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        self.assertGreaterEqual(accuracy, accuracy_threshold)
        self.assertGreaterEqual(f1, f1_threshold)
        return accuracy, f1

class TestModels(GeneralizedTest):
    
    def __init__(self, file_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_path = file_path

    def test_nb(self):
        self.test_training(GaussianNB, self.file_path)
        self.test_prediction(GaussianNB, self.file_path)
        self.test_accuracy(GaussianNB, self.file_path)

    def test_ann(self):
        def ann_model(input_shape):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=64, activation='relu', input_shape=input_shape),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model

        df = self.load_data(self.file_path)
        X_train, X_test, y_train, y_test = self.traintest_split(df)
        input_shape = [X_train.shape[1]]

        model = ann_model(input_shape)
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=256, epochs=60, callbacks=[earlystopping], verbose=0)
        
        self.assertIsNotNone(model)
        predictions = (model.predict(X_test) > 0.5).astype("int32")
        self.assertEqual(len(predictions), len(X_test))
        
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        self.assertGreaterEqual(accuracy, 0.9)
        self.assertGreaterEqual(f1, 0.9)

    def test_svm(self):
        self.test_training(SVC, self.file_path, {'random_state': 42})
        self.test_prediction(SVC, self.file_path, {'random_state': 42})
        self.test_accuracy(SVC, self.file_path, {'random_state': 42})

    def test_logreg(self):
        self.test_training(LogisticRegression, self.file_path, {'random_state': 42, 'solver': 'sag'})
        self.test_prediction(LogisticRegression, self.file_path, {'random_state': 42, 'solver': 'sag'})
        self.test_accuracy(LogisticRegression, self.file_path, {'random_state': 42, 'solver': 'sag'})

    def test_rfc(self):
        self.test_training(RandomForestClassifier, self.file_path, {'random_state': 42})
        self.test_prediction(RandomForestClassifier, self.file_path, {'random_state': 42})
        self.test_accuracy(RandomForestClassifier, self.file_path, {'random_state': 42})

    def test_dtc(self):
        self.test_training(DecisionTreeClassifier, self.file_path, {'random_state': 42})
        self.test_prediction(DecisionTreeClassifier, self.file_path, {'random_state': 42})
        self.test_accuracy(DecisionTreeClassifier, self.file_path, {'random_state': 42})

def main(file_path):
    suite = unittest.TestSuite()
    test_classes = [TestModels]

    for test_class in test_classes:
        test_loader = unittest.TestLoader()
        tests = test_loader.loadTestsFromTestCase(test_class)
        for test in tests:
            test._testMethodName = test._testMethodName
            test.file_path = file_path
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=4)
    runner.run(suite)

if __name__ == '__main__':
    file_path = '' #load in dataset  
    main(file_path)
