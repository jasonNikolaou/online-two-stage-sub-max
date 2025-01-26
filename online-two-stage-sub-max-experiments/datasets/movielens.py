import numpy as np
import pandas as pd
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from WTP import Potential, WTP

n = 2000 # movies
m = 100
nr = 20

userRatingsdf = pd.read_csv('./datasets/ratings.csv', sep='\t')  # Replace 'file_path.csv' with the actual file path
# Convert each row to a list and print
userRatings = []
for _, row in userRatingsdf.iterrows():
    userRatings.append(row.tolist())

# Ratings are in the format - userId, movieId, rating, timestamp
print(f'user rating example =  {userRatings[0]}')

ratings = dict()

for r in userRatings:
    movId = r[1]
    rating = r[2]
    
    if movId in ratings:
        ratings[movId].append(rating)
    else:
        ratings[movId] = [rating]

print(f'# of movies = {len(ratings)}')

# assign a positive rating only to movies that have at least nr users rating them
for r in ratings:
    if len(ratings[r]) >= nr:
        ratings[r] = np.mean(ratings[r])
    else:
        ratings[r] = 0

print(f'ratings examples = {ratings[1], ratings[42]}')

# Pick the movies with the best n ratings
lowest = sorted(ratings.values())[len(ratings.values()) - n - 1]
movies = [movId for movId in ratings if ratings[movId] > lowest]
print(f'Some of the best rated movies = {movies[:10]}') 

# Get the users who rated these movies. From their list, pick the m of them that rated most movies
numRatings = dict()
for r in userRatings:
    userID = r[0]
    mov = r[1]
    
    if mov in movies:
        if userID in numRatings:
            numRatings[userID] += 1
        else:
            numRatings[userID] = 0
    
# Sort users by the number of ratings in descending order
sorted_users = sorted(numRatings.items(), key=lambda x: x[1], reverse=True)

# Pick the top m users
users = [user for user, count in sorted_users[:m]]

print('Number of users:', len(users))
print('Some user ids..', users[:5])
print('Number of movies', len(movies))
print('Some movie ids..', movies[:5])

userMoviesRatings = dict() # keys: (user, movie), value: rating
final_users = set()
final_movies = set()
for r in userRatings:
    usr = r[0]
    mov = r[1]
    rating = r[2]
    
    if (usr in users) and (mov in movies):
        userMoviesRatings[(usr, mov)] = rating

for user in users:
    for mov in movies:
        if (user, mov) not in userMoviesRatings:
            userMoviesRatings[user, mov] = 0

print(f'final number of users = {len(users)}')
print(f'final number of movies = {len(movies)}')

print('Converting to WTP...')

n = len(movies)
wtps = []

for v_prime in users:
    potentials = []
    weights = []
    # Extract weights for all v with respect to v'
    w_v_prime = np.array([userMoviesRatings[(v_prime, v)] for v in movies])

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

    potentials = [potentials[i] for i in range(len(weights)) if weights[i] > 0]
    weights = [weight for weight in weights if weight > 0]
    
    # Create the WTP object
    wtps.append(WTP(potentials=potentials, weights=weights))

file = './instances/movies.pkl'
with open(file, 'wb') as file:
    pickle.dump(wtps, file)
print(f"List saved to {file}")
