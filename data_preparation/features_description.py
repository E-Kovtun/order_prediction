# All categorical features
categorical_features = ['Ship.to', 'PLZ', 'Day', 'Month', 'Material', 'Status',
                        'MaterialGroup.1', 'MaterialGroup.2', 'MaterialGroup.4']

# Features that are processed with Bag of Words method
features_for_vectorizer = ['Material', 'MaterialGroup.1', 'MaterialGroup.2', 'MaterialGroup.4']

# Features that are constant within the time for a particular restaurant
constant_features_one_hot = ['Ship.to', 'PLZ']

# Features that are changing withing the time for a particular restaurant
changing_features_one_hot = ['Day', 'Month', 'Status']

numerical_features = ['dt', 'Amount_HL']

all_features = categorical_features + numerical_features
