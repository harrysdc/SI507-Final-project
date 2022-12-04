# Book Recommendation System

This is a SI507 final project designed to recommend and present books to users based on user preference

## Data sources

The project relies on:
* [The Google Books API](https://developers.google.com/books)

## Getting started

### Prerequisites

Python3, requests, flask, scikit-learn, matplotlib, domininate

```
$ conda create -p test-env python=3.7
$ conda activate .\test-env
$ conda install -r requirements.txt
$ python getbooks.py
```

## Data presentation
Once the Flask application is running, a user can navigate to “gallery.html” or http://127.0.0.1:5000/ to see the recommended books
