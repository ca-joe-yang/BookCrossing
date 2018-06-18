Book Crossing
===
- https://learner.csie.ntu.edu.tw/judge/ml18spring/track1/

# Errors
- "02e87fe603,014028009,8", "02e87fe603,014028009X,8"
- Books ISBN
    + Books in "book_ratings_train.csv" but not in "books.csv": 23262
    + Books in "book_ratings_test.csv" but not in "books.csv": 16307
    + Books in "implicit_ratings.csv" but not in "books.csv": 44925

# Requirements
- wget

# Inputs
- User
    + Location (City / State / Country)
    + Age
- Book
    + ISBN
    + Title
    + Authors
    + Year
    + Publisher
    + Image
    + Description

# Outpus
- Rating: 0 ~ 10
