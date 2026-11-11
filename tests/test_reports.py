from house_purchase.report import metrics_at_threshold

def test_metrics_shape():
    scores = metrics_at_threshold([0,0,1,1],[.01,.2,.7,.9])
    assert scores['accuracy'] == 1.0
    assert scores['confusion_matrix'] == [[2,0],[0,2]]
    assert scores['average_precision'] == 1.0
