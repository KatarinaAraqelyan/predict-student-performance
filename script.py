import math
from collections import Counter, defaultdict
from functools import partial
import random
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load and preprocess the data
inputs = []

remove = ['gender', 'nationality', 'placeofbirth', 'sectionid', 'topic',
          'semester', 'relation', 'parentschoolsatisfaction']
convert = ['raisedhands', 'visitedresources', 'announcementsview', 'discussion']

with open('api-edu-data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    keys = []
    for row in csv_reader:
        if line_count == 0:
            keys = [x.lower() for x in row]
            print(keys)
        else:
            new_dict = {}
            c = 0
            for key in keys:
                if key not in remove:
                    if key in convert:
                        val = int(row[c])
                        if val <= 33:
                            m = "L"
                        elif val <= 66:
                            m = "M"
                        else:
                            m = "H"
                        new_dict[key] = m
                    else:
                        new_dict[key] = row[c]
                c += 1

            grade = new_dict['class'] == "H"
            new_dict.pop('class')
            inputs.append((new_dict, grade))
        line_count += 1

# Helper functions for decision tree
def entropy(class_probabilities):
    return sum(-p * math.log(p, 2) for p in class_probabilities if p)

def class_probabilities(labels):
    return [count / len(labels) for count in Counter(labels).values()]

def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

def partition_by(inputs, attribute):
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute]
        groups[key].append(input)
    return groups

def partition_entropy(subsets):
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * len(subset) / total_count for subset in subsets)

def partition_entropy_by(inputs, attribute):
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())

def classify(tree, input):
    if tree in [True, False]:
        return tree
    attribute, subtree_dict = tree
    subtree_key = input.get(attribute)
    if subtree_key not in subtree_dict:
        subtree_key = None
    subtree = subtree_dict[subtree_key]
    return classify(subtree, input)

def build_tree(inputs, num_split_candidates=2, split_candidates=None):
    if split_candidates is None:
        split_candidates = list(inputs[0][0].keys())

    num_trues = len([label for _, label in inputs if label])
    num_falses = len(inputs) - num_trues

    if num_trues == 0: return False
    if num_falses == 0: return True
    if not split_candidates: return num_trues >= num_falses

    if len(split_candidates) <= num_split_candidates:
        sampled_split_candidates = split_candidates
    else:
        sampled_split_candidates = random.sample(split_candidates, num_split_candidates)

    best_attribute = min(sampled_split_candidates, key=partial(partition_entropy_by, inputs))
    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates if a != best_attribute]

    subtrees = {attr_val: build_tree(subset, split_candidates=new_candidates)
                for attr_val, subset in partitions.items()}
    subtrees[None] = num_trues > num_falses
    return (best_attribute, subtrees)

def forest_classify(trees, input):
    votes = [classify(tree, input) for tree in trees]
    vote_counts = Counter(votes)
    return "High Performance" if vote_counts.most_common(1)[0][0] else "Low Performance"

# Split testing and training data
testing = [inputs.pop(random.randint(0, len(inputs)-1)) for _ in range(15)]
all_trees = [build_tree(random.sample(inputs, 200)) for _ in range(50)]

# Example student predictions
student_1 = {'stageid': 'MiddleSchool', 'raisedhands': 'L', 'visitedresources': 'M',
             'announcementsview': 'L', 'discussion': 'L', 'parentansweringsurvey': 'Yes',
             'studentabsencedays': 'Over-7'}

student_2 = {'visitedresources': 'H', 'announcementsview': 'H', 'discussion': 'H',
             'parentansweringsurvey': 'Yes', 'studentabsencedays': 'Under-7'}

print("Student 1 has " + forest_classify(all_trees, student_1))
print("Student 2 has " + forest_classify(all_trees, student_2))

# Accuracy on test set
good = 0
total = 0
for i in testing:
    pred = forest_classify(all_trees, i[0])
    actual = "High Performance" if i[1] else "Low Performance"
    if pred == actual:
        good += 1
    total += 1

accuracy = good / total * 100
print("The random forest's accuracy is " + str(accuracy) + "%")

# Confusion Matrix
y_true = ["High Performance" if label else "Low Performance" for _, label in testing]
y_pred = [forest_classify(all_trees, x) for x, _ in testing]

cm = confusion_matrix(y_true, y_pred, labels=["High Performance", "Low Performance"])
disp = ConfusionMatrixDisplay(cm, display_labels=["High Performance", "Low Performance"])
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.show()

# Feature importance
def gather_attributes(tree):
    if tree in [True, False]:
        return []
    attr, subtrees = tree
    attrs = [attr]
    for t in subtrees.values():
        attrs.extend(gather_attributes(t))
    return attrs

all_attributes = []
for tree in all_trees:
    all_attributes.extend(gather_attributes(tree))

importance = Counter(all_attributes)
plt.figure(figsize=(10,5))
plt.bar(importance.keys(), importance.values())
plt.xticks(rotation=45)
plt.ylabel("Times chosen as split")
plt.title("Feature Importance in Random Forest")
plt.show()

# Feature distributions
categorical_features = ['raisedhands', 'visitedresources', 'announcementsview', 'discussion']

for col in categorical_features:
    counts = Counter(row[0][col] for row in inputs + testing)
    plt.figure()
    plt.bar(counts.keys(), counts.values())
    plt.title(f"Distribution of {col}")
    plt.ylabel("Number of students")
    plt.show()

# Random Forest Votes for Example Students
for student, name in [(student_1, "Student 1"), (student_2, "Student 2")]:
    votes = [classify(tree, student) for tree in all_trees]
    votes_count = Counter(votes)
    plt.bar(['High', 'Low'], [votes_count[True], votes_count[False]])
    plt.title(f"Random Forest Votes for {name}")
    plt.show()
