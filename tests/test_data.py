import numpy as np
import pandas as pd
from house_purchase.config import FEATURES
from house_purchase.data import split_data, make_preprocessor, training_input_schema, load_data


def test_stratification_and_split(sample_data):
    splits = split_data(sample_data)
    assert (len(splits.y_train),len(splits.y_val),len(splits.y_test)) == (70,10,20)
    assert len(set(splits.X_train.index)&set(splits.X_val.index))==0
    assert len(set(splits.X_train.index)&set(splits.X_test.index))==0
    assert list(splits.X_train.columns) == FEATURES
    assert splits.y_train.mean() == .5


def test_preprocessor_fits_only_training_categories(sample_data):
    splits=split_data(sample_data)
    pre=make_preprocessor()
    transformed=pre.fit_transform(splits.X_train)
    assert transformed.shape[0] == 70
    # Unseen categories must not cause a crash or expand the output dimension.
    unseen = splits.X_val.copy()
    unseen.loc[unseen.index[0], "city"] = "UNKNOWN_TEST_CITY"
    assert pre.transform(unseen).shape[1] == transformed.shape[1]


def test_validation_columns_and_target(tmp_path, sample_data):
    target=tmp_path/'fake.csv'
    sample_data.to_csv(target,index=False)
    assert len(load_data(target)) == 100
    target.write_text('decision,wrong\n1,2\n', encoding='utf-8')
    import pytest
    with pytest.raises(ValueError, match='Missing columns'):
        load_data(target)


def test_schema_contains_no_rows(sample_data):
    d=training_input_schema(sample_data)
    assert len(d['numerical']) == 19
    assert len(d['categorical']) == 4
    assert 'property_id' not in d['feature_order']
    assert 'decision' not in d['feature_order']
