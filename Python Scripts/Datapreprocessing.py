from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Fill missing age values with the median age
imputer = SimpleImputer(strategy='median')
train_data['Age'] = imputer.fit_transform(train_data[['Age']])
test_data['Age'] = imputer.transform(test_data[['Age']])

# Convert 'Sex' to numerical values
encoder = LabelEncoder()
train_data['Sex'] = encoder.fit_transform(train_data['Sex'])
test_data['Sex'] = encoder.transform(test_data['Sex'])

# Select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X_train = train_data[features]
y_train = train_data['Survived']
X_test = test_data[features]