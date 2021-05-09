from src.data.make_dataset import split_train_val_data, extract_target, read_data
from src.entities import SplittingParams, FeatureParams
from tests.data_generator import generate_dataset


def test_split_dataset():
    test_size = 0.2
    splitting_params = SplittingParams(random_state=1, test_size=test_size)
    feature_params = FeatureParams(categorical_features=["cp", "restecg", "slope", "ca", "thal"],
                                   numerical_features=["age", "trestbps", "chol", "thalach", "oldpeak"],
                                   target_col='target')
    data = generate_dataset(100)
    data, target = extract_target(data, feature_params)
    X_train, y_train, X_test, y_test = split_train_val_data(data, target, splitting_params)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_train.shape[0] > 10
    assert X_test.shape[0] == y_test.shape[0]
    assert X_test.shape[0] > 10
