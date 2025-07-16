# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from io import StringIO

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "SujalPatidar24"
        repo_name = "MLOps-Capstone"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the new model from MLflow model registry
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Load the vectorizer
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        mock_csv = """ review,sentiment
        "Film version of Sandra Bernhard's one-woman off-Broadway show is gaspingly pretentious. Sandra spoofs lounge acts and superstars, but her sense of irony is only fitfully interesting, and fitfully funny. Her fans will say she's scathingly honest, and that may be true. But she's also shrill, with an unapologetic, in-your-face bravado that isn't well-suited to a film in this genre. She doesn't want to make nice--and she's certainly not out to make friends--and that's always going to rub a lot of people the wrong way. But even if you meet her halfway, her material here is seriously lacking. Filmmaker Nicolas Roeg served as executive producer and, though not directed by him, the film does have his chilly, detached signature style all over it. Bernhard co-wrote the show with director John Boskovich; their oddest touch was in having all of Sandra's in-house audiences looking completely bored--a feeling many real viewers will most likely share. *1/2 from ****",negative
        I switched this on (from cable) on a whim and was treated to quite a surprise...although very predictable this film turned out to be quite enjoyable...no big stars but well-directed and just plain fun. With all the over-hyped crap that is out there it is very nice to get an unexpected surprise now and then... and this little film fits the bill nicely. 9/10,positive
        "The `plot' of this film contains a few holes you could drive a massive truck through, but I reckon that isn't always top priority in horror. Two elderly sisters in rural England keep their brother in the cellar since more than 30 years. Now, he escaped and started a killing spree, focusing on militaries that are homed nearby. `We only did we thought was best for him' they keep on repeating and  strangely  all the army officers love these women and don't doubt their sincerity, even though 5 of their men died. I don't know whether to find the revelation near the end suspenseful  or tedious! In a way, this film reminded me about `Arsenic and Old Lace'. In that black-comedy classic, two half-insane siblings mother their goofy younger brother as well, yet they do the killing there. The old ladies in `The Beast in the Cellar' are by no means less crazy, though. The `horror' in this early 70's film is very amateurish and cheap, but there are a few neat attempts to build up the tension. Too many `old-ladies' talk about the good ol' days, though and that rarely is something you seek in a horror film with such an appealing title. Flora Robson, who may be recognized by classic film buffs, plays one of the sisters. She gave image to the Queen of England is the legendary Errol Flynn swashbuckler film, the Sea Hawk.",negative
        "Some amusing humor, some that falls flat, some decent acting, some that is quite atrocious. This movie is simply hit and miss, guaranteed to amuse 12 year old boys more than any other niche.<br /><br />The child actors in the movie are just unfunny. When you are making a family comedy, that does tend to be a problem. Beverly D'Angelo rises above the material to give a funny, and dare I say it, human performance in the midst of this mediocrity.",negative"""
        cls.holdout_data = pd.read_csv(StringIO(mock_csv))

        # Use with open to avoid ResourceWarning
        with open('models/vectorizer.pkl', 'rb') as f:
            cls.vectorizer = pickle.load(f)

        # Load holdout test data
        # cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Create a dummy input for the model based on expected input shape
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])

        # Predict using the new model to verify the input and output shapes
        prediction = self.new_model.predict(input_df)

        # Verify the input shape
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))

        # Verify the output shape (assuming binary classification with a single output)
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)  # Assuming a single output column for binary classification

    def test_model_performance(self):
        # Extract features and labels from holdout test data
        X_holdout = self.holdout_data.iloc[:,0:-1]
        y_holdout = self.holdout_data.iloc[:,-1]

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 score should be at least {expected_f1}')

if __name__ == "__main__":
    unittest.main()