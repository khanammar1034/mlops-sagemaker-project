from sklearn.datasets import load_iris


def test_dataset_loads():
    iris = load_iris()

    assert len(iris.data) > 0
    assert len(iris.target) > 0