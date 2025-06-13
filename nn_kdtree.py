import numpy as np
import pandas as pd 
import sys

STUDENT_ID = 'a1884774' # student ID
DEGREE = 'UG' # undagraduwate

# me tink me gwaan load de data
def load_data(train_path, test_path):
    # read data of the training
    kumalala = pd.read_csv(train_path, sep='\s+', header=0)
    X_train = kumalala.iloc[:, :-1].to_numpy()
    y_train = kumalala.iloc[:, -1].to_numpy()

    # read da test data
    savesta = pd.read_csv(test_path, sep='\s+', header=0)
    X_test = savesta.to_numpy()

    return X_train, y_train, X_test

# kaydee node
class KDNode:
    # its tuesday init
    def __init__(self, point=None, label=None, split_dim=None, split_val=None, left=None, right=None):
        self.point = point
        self.label = label
        self.split_dim = split_dim
        self.split_val = split_val
        self.left = left
        self.right = right

# Reverse onceler on the (kd) trees (recursive function)
def build_KDtree(points, labels, depth=0, is_root=None):
    if len(points) == 0:
        return None
    if len(points) == 1:
        return KDNode(point=points[0], label=labels[0])
    
    waduh = points.shape[1] # number of dihmensitons
    axis = depth % waduh

    # use numpy argsort thing to sort using axis
    spongebob = points[:, axis].argsort()
    points = points[spongebob]
    labels = labels[spongebob]

    medihan_idx = len(points) // 2
    medihan_point = points[medihan_idx]
    medihan_label = labels[medihan_idx]
    medihan_val = medihan_point[axis]

    # use median point as the current node
    node = KDNode(
        point=medihan_point,
        label=medihan_label,
        split_dim=axis,
        split_val=medihan_val
    )

    # ts does NOT work
    # left_points = points[:medihan_idx]
    # left_labels = labels[:medihan_idx]

    # # we exlcude the medihan becase yeah
    # right_points = points[medihan_idx + 1:]
    # right_labels = labels[medihan_idx + 1:]

    # my id is even steven, so we "equal to or" as the assignment said
    # split manually, excluding the median point
    left_points = []
    left_labels = []
    right_points = []
    right_labels = []

    # ts also does NOT work
    # # build the sub(scribe)trees recursively
    # left_child = build_KDtree(left_points, left_labels, depth + 1, is_root=False)
    # right_child = build_KDtree(right_points, right_labels, depth + 1, is_root=False)


    # # print split info for first level just to double check for errors (subtle foreshadowing)
    # if is_root:
    #     print("." * (depth + 1) + f"l{len(left_points)}")
    #     print("." * (depth + 1) + f"r{len(right_points)}")

    for i in range(len(points)):
        if i == medihan_idx:
            continue  # uno skip the median point 
        if points[i][axis] < medihan_val:
            left_points.append(points[i])
            left_labels.append(labels[i])
        elif points[i][axis] > medihan_val:
            right_points.append(points[i])
            right_labels.append(labels[i])
        else:
            # if equal to median, place in right subtree
            right_points.append(points[i])
            right_labels.append(labels[i])

    left_points = np.array(left_points)
    right_points = np.array(right_points)
    left_labels = np.array(left_labels)
    right_labels = np.array(right_labels)

    if is_root:
        print("." * (depth + 1) + f"l{len(left_points)}")
        print("." * (depth + 1) + f"r{len(right_points)}")

    node.left = build_KDtree(left_points, left_labels, depth + 1, is_root=False)
    node.right = build_KDtree(right_points, right_labels, depth + 1, is_root=False)

    return node

    # return KDNode(
    #     point=points[medihan_idx],
    #     label=labels[medihan_idx],
    #     split_dim=axis,
    #     split_val=medihan_val,
    #     left=left_child,
    #     right=right_child
    # )

    

def main():
    kumalala = sys.argv[1]
    savesta = sys.argv[2]
    chesta = int(sys.argv[3])

    X_train, y_train, X_test = load_data(kumalala, savesta)

    print(f"train data: {X_train.shape}, Labels: {y_train.shape}")
    print(f"test data: {X_test.shape}")

    # build kevin durant trees (with the even split)
    lorax = build_KDtree(X_train, y_train, depth=chesta, is_root=True)
    
    # 1-nn goes here after


if __name__ == "__main__":
    main()