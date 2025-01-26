import os
import numpy as np
import xml.etree.ElementTree as ET
import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from WTP import Potential, WTP

# Parameters
n = 150 # total images
annotation_dir = "./datasets/VOC2012/Annotations"

# Initialize variables
images = {}
categories = set()

# Parse annotations
print("Parsing annotations...")
cnt = 0
for file in os.listdir(annotation_dir):
    if file.endswith(".xml"):
        cnt += 1
        if cnt > n:
            break

        file_path = os.path.join(annotation_dir, file)
        tree = ET.parse(file_path)
        root = tree.getroot()

        cur_file = root.find("filename").text
        images[cur_file] = []

        for obj in root.findall("object"):
            category = obj.find("name").text
            images[cur_file].append(category)
            categories.add(category)

categories = sorted(categories)
print(categories)
print(f"Number of categories: {len(categories)}")

# Map categories to images
cat_img = {cat: [] for cat in categories}
for img, cat_list in images.items():
    for cat in set(cat_list):  # Avoid duplicates
        cat_img[cat].append(img)

print(f"Sample images for category 'sheep': {cat_img['sheep'][:5]}")

# Create feature vectors
print("Creating feature vectors...")
featvec = {}
for img, cat_list in images.items():
    vec = np.zeros(len(categories), dtype=int)
    for cat in cat_list:
        vec[categories.index(cat)] += 1
    featvec[img] = vec


dist = dict()
for el1 in featvec:
    for el2 in featvec:
        dist[(el1, el2)] = np.linalg.norm(featvec[el1] - featvec[el2])

dist = {
    (img1, img2): np.linalg.norm(vec1 - vec2)
    for img1, vec1 in featvec.items()
    for img2, vec2 in featvec.items()
}

sim = dict()
max_distance = max(dist.values())
for key, d in dist.items():
    sim[key] = (1 - d / max_distance) * 100
    # sim[key] = np.exp(-0.1 * d)
    # print(sim[key], dist[key])

print("Processing complete.")


print('Converting to WTP...')

n = len(images)
wtps = []

for v_prime in images:
    potentials = []
    weights = []
    # Extract weights for all v with respect to v'
    w_v_prime = np.array([sim[(v, v_prime)] for v in images])

    # Get the permutation π such that w_π1 >= w_π2 >= ... >= w_πn
    permutation = np.argsort(-w_v_prime)  # Descending order
    permuted_weights = w_v_prime[permutation]  # Permuted weights

    # Create n potentials for the current v_prime
    for i in range(n):
        # Define weight vector for the i-th potential
        w_i = np.zeros(n)
        w_i[permutation[:i + 1]] = 1  # Only the top i+1 elements are 1

        # Create the potential with b = 1 and the permuted weights
        potential = Potential(b=1, w=w_i)
        potentials.append(potential)

        # Weight for the WTP is the difference (w_πi,v' - w_π(i+1),v')
        if i < n - 1:
            weights.append(permuted_weights[i] - permuted_weights[i + 1])
        else:
            weights.append(permuted_weights[i])  # Last term has no successor

    # Normalize weights by 1 / |V|
    weights = [w / n for w in weights]

    # Create the WTP object
    wtps.append(WTP(potentials=potentials, weights=weights))

m = 10 # number of different functions
wtps = wtps[:m]
file = './instances/images.pkl'
with open(file, 'wb') as file:
    pickle.dump(wtps, file)
print(f"List saved to {file}")
