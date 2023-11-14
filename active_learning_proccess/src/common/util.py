import numpy as np
def compute_class_weight(classes, y):
    weight = []

    for x in classes:
        value_class = y.count(x)
        weight.append((len(y)- value_class)/value_class)
    return weight

# return [(len(y)- y.count(x))/y.count(x) for x in classes]
        

    