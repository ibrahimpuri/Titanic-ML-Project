import matplotlib.pyplot as plt
import seaborn as sns

# Check for missing values
print(train_data.isnull().sum())

# Survival rate
print(f"Survival Rate: {train_data['Survived'].mean()*100:.2f}%")

# Visualize the survival rate by gender
sns.barplot(x='Sex', y='Survived', data=train_data)
plt.show()

# Visualize the survival rate by passenger class
sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.show()