# part4_visualization_ml.py
# Assignment 3 - Part 4
# analysing student performance and trying to predict pass/fail

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # added this because plots were not showing earlier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ==================================================
# Task 1: data loading and basic checking
# ==================================================

print("starting program...")

# reading csv file

df = pd.read_csv("students.csv")

# used this earlier to check everything
# print(df)

print("\nshowing first few rows just to confirm data:")
print(df.head())
print("\nshape of dataset:")
rows, cols = df.shape
print("rows =", rows, "columns =", cols)
print("\ndata types of each column:")
print(df.dtypes)
print("\nbasic stats:")
print(df.describe())

# checking missing values (probably none but still checking)

print("\nnull values check:")
nulls = df.isnull().sum()
print(nulls)
print("\npass/fail count:")
counts = df['passed'].value_counts()
print(counts)


# subjects list

subs = ['math', 'science', 'english', 'history', 'pe']

# calculating average scores (did this step separately so easier to debug)

avg_list = df[subs].mean(axis=1)
df['avg_score'] = avg_list

# also calculating overall avg again (kind of same thing but leaving it)

df['overall_avg'] = df[subs].mean(axis=1)


print("\naverage marks (passed students):")
p1 = df[df['passed'] == 1][subs].mean()
print(p1)

print("\naverage marks (failed students):")
p0 = df[df['passed'] == 0][subs].mean()
print(p0)


# finding top student

max_index = df['overall_avg'].idxmax()
top = df.loc[max_index]

print("\ntop student based on avg:")
print(top['name'], "->", top['overall_avg'])


# ==================================================
# Task 2: matplotlib graphs
# ==================================================

# bar chart
plt.figure()

means = df[subs].mean()
means.plot(kind='bar')

plt.title("Average Score per Subject")
plt.xlabel("Subjects")
plt.ylabel("Score")
plt.savefig("plot_bar.png")
plt.show()


# histogram 
plt.figure()
plt.hist(df['math'], bins=5)

mean_val = df['math'].mean()

plt.axvline(mean_val, linestyle='dashed')
plt.title("Math Scores")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.savefig("plot_hist.png")
plt.show()


# scatter plot
plt.figure()

# splitting into two groups

passed_df = df[df['passed'] == 1]
failed_df = df[df['passed'] == 0]

# plotting both

plt.scatter(passed_df['study_hours_per_day'], passed_df['avg_score'], label="Pass")
plt.scatter(failed_df['study_hours_per_day'], failed_df['avg_score'], label="Fail")
plt.title("Study Hours vs Avg Score")
plt.xlabel("Study Hours")
plt.ylabel("Avg Score")
plt.legend()
plt.savefig("plot_scatter.png")
plt.show()

# box plot

plt.figure()

att1 = df[df['passed'] == 1]['attendance_pct']
att0 = df[df['passed'] == 0]['attendance_pct']

plt.boxplot([att1, att0], tick_labels=['Pass', 'Fail'])
plt.title("Attendance comparison")
plt.ylabel("Attendance %")
plt.savefig("plot_box.png")
plt.show()


# line plot 
plt.figure()
plt.plot(df['name'], df['math'], marker='o', label='Math')
plt.plot(df['name'], df['science'], marker='x', label='Science')

# labels overlap otherwise

plt.xticks(rotation=45)
plt.title("Math vs Science")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.legend()
plt.savefig("plot_line.png")
plt.show()


# ==================================================
# Task 3: seaborn graphs
# ==================================================

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.barplot(data=df, x='passed', y='math')
plt.title("Math vs Pass")
plt.subplot(1,2,2)
sns.barplot(data=df, x='passed', y='science')
plt.title("Science vs Pass")
plt.savefig("plot_seaborn_bar.png")
plt.show()
plt.figure()
sns.scatterplot(data=df, x='attendance_pct', y='avg_score', hue='passed')

# adding regression lines separately
sns.regplot(data=df[df['passed'] == 1], x='attendance_pct', y='avg_score', scatter=False)
sns.regplot(data=df[df['passed'] == 0], x='attendance_pct', y='avg_score', scatter=False)
plt.title("Attendance vs Avg")
plt.savefig("plot_seaborn_scatter.png")
plt.show()


# quick note:
# seaborn is easier for grouped stuff
# matplotlib gives more control but more work


# ==================================================
# Task 4: machine learning part
# ==================================================

# selecting features

cols_for_model = ['math','science','english','history','pe','attendance_pct','study_hours_per_day']
X = df[cols_for_model]

# target
y = df['passed']
print("\nsplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaling (important step)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("training model...")
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
train_score = model.score(X_train_scaled, y_train)
print("training accuracy:", train_score)

# predictions
pred = model.predict(X_test_scaled)
test_score = accuracy_score(y_test, pred)
print("test accuracy:", test_score)

print("\nchecking predictions:")

names = df.loc[X_test.index, 'name']

for n, a, p in zip(names, y_test, pred):
    if a == p:
        print(n, "- correct")
    else:
        print(n, "- wrong")


# ==================================================
# feature importance
# ==================================================

coef = model.coef_[0]
feat_names = list(X.columns)
imp = pd.DataFrame()
imp['Feature'] = feat_names
imp['Coefficient'] = coef
imp['abs'] = imp['Coefficient'].abs()
imp = imp.sort_values(by='abs', ascending=False)
print("\nfeature importance:")
print(imp[['Feature','Coefficient']])

plt.figure()
colours = ['green' if x > 0 else 'red' for x in imp['Coefficient']]
plt.barh(imp['Feature'], imp['Coefficient'], color=colours)
plt.title("Feature Importance")
plt.xlabel("Value")
plt.savefig("feature_importance.png")
plt.show()


# ==================================================
# bonus prediction
# ==================================================

print("\npredicting new student...")
new_data = [[75,70,68,65,80,82,3.2]]
scaled_new = scaler.transform(new_data)
res = model.predict(scaled_new)
prob = model.predict_proba(scaled_new)

if res[0] == 1:
    print("Result: Pass")
else:
    print("Result: Fail")

print("Probability:", prob)