import csv
import json
import requests
import time
from flask import Flask, render_template
from sklearn import tree
import matplotlib.pyplot as plt
from dominate import document
from dominate.tags import *
from data_structure import Node


class Book:
    def __init__(self, json=None):
        '''
        year, pageCount, categories
        '''
        if json is not None:
            if "volumeInfo" in json:
                self.title = json["volumeInfo"]["title"]
                self.language = json["volumeInfo"]["language"]

                try:
                    self.release_year = json["volumeInfo"]["publishedDate"][0:4]
                except:
                    self.release_year = "No Release Year"

                try:
                    self.author = json["volumeInfo"]["authors"][0]
                except:
                    self.author = "No Author"

                try:
                    self.pageCount = json["volumeInfo"]["pageCount"]
                except:
                    self.pageCount = "No PageCount"

                try:
                    self.categories = json["volumeInfo"]["categories"][0]
                except:
                    self.categories = "No Categories"
                
                try:
                    self.imageLinks = json["volumeInfo"]['imageLinks']["thumbnail"]
                except:
                    self.imageLinks = "No Image"
                try:
                    self.previewLink = json["volumeInfo"]["previewLink"]
                except:
                    self.previewLink = "No previewLink"

    def info(self):
        return self.title + ' by ' +self.author + ' (' + str(self.release_year) + ')'

    def length(self):
        return 0
    
    def getTrainData(self, user_short, user_category, user_recent):
        '''
        convert text entry to number
        '''
        # convert category string to digit for training
        if self.categories == "Cooking":
            category_num = 1
        elif self.categories == "Biography & Autobiography":
            category_num = 2
        elif self.categories == "Business & Economics":
            category_num = 3
        else:
            category_num = 4
        
        if self.pageCount == "No PageCount":
            pageCnt = -1
        else: 
            pageCnt = int(self.pageCount)
        
        if self.release_year == "No Release Year":
            rYear = -1
        else:
            try:
                rYear = int(self.release_year)
            except:
                rYear = -1
            
        

        # label data
        if user_short == 'y':
            user_short = True
        else:
            user_short = False
        if user_recent == 'y':
            user_recent = True
        else:
            user_recent = False

        if (user_short==(pageCnt<200)) and (user_recent==(rYear>=2012)) and ((user_category[0]=='y')==(category_num==1)) and ((user_category[1]=='y')==(category_num==2)):
            return [rYear, category_num, pageCnt], [self.title, self.author, self.categories, self.imageLinks, rYear, self.previewLink], 1
            
        else:
            return [rYear, category_num, pageCnt], [self.title, self.author, self.categories, self.imageLinks, rYear, self.previewLink], 0



def getBooks(title):
    GB_URL = "https://www.googleapis.com/books/v1/volumes?q=" + title + "&maxResults=30"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36"}

    response = requests.get(GB_URL, headers=headers)
    response_json = json.loads(response.text)
    result_list = response_json['items']

    books_list = []
    for result in result_list:
        if "title" in result["volumeInfo"]:
            books_list.append(Book(json=result))
    
    return books_list


def train_decision_tree(X, y, features = ['release_year', 'category', 'pageCnt']):
    '''
    https://www.w3schools.com/python/python_ml_decision_tree.asp
    '''

    print('Training decision tree model with 300 data...')
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    tree.plot_tree(clf, feature_names=features)
    plt.savefig('photos/booksDecisionTree.png')
    return clf


def predict_decision_tree(clf, test_x, test_info, features = ['release_year', 'category', 'pageCnt']):
    print('Predicting with decision tree model...')
    pred = clf.predict(test_x)

    preds_info = []
    for i in range(len(pred)):
        if pred[i]==1:
            preds_info.append(test_info[i])
    return preds_info

       
def getLibrary(user_short, user_category, user_recent, mode='train'):
    '''
    get books for 200 entry
    label them: <150 page, >2000 year , category=Biography & Autobiography
    turn category into number
    '''
    print('Fetching data from Google Books API...')

    name_train = ["donald trump", "mariah carey",  "japanese recipes", "steve jobs", "adam grant", "Harry Potter", "cooking", "entrepreneur", "Neil Gaiman", "dale carnegie"]
    name_test = ["joe biden", "Gordon Ramsay", "elon musk", "Simon Sinek", "toyota", "Vaclav Smil", "semiconductor", "Stephen King", "cookbook", "MBA", "chef"]

    if mode == 'train':
        names = name_train
    else:
        names = name_test

    train_list = []
    for name in names:
        books_list = getBooks(name)
        train_list.extend(books_list)
    cnt = 0
    train_X = []
    train_y = []
    train_info = []
    for book in train_list:
        X, info, y = book.getTrainData(user_short, user_category, user_recent)
        train_X.append(X)
        train_info.append(info)
        train_y.append(y)

    return train_X, train_y, train_info



def write_html(title_list, author_list, category_list, link_list, year_list, preview_list, DISPLAY_TREE='y'):
    '''
    https://github.com/Knio/dominate
    '''

    import glob
    photos = glob.glob('photos/*.png')
    
    with document(title='Photos') as doc:
        link(rel='stylesheet', href='style.css')

        # generate template for decision tree
        if DISPLAY_TREE == 'y':
            h1('Your Decision Tree')
            for path in photos:
                div(img(src=path), _class='photo')

        # generate template for images
        h1('Your Recommended Books Powered by Google Books API and Decision Tree')
        with table().add(tbody()):
            l = tr()
            l += th('No.')
            l += th('Thumbnail')
            l += th('Description')
            l += th('Preview Link')
            for i in range(len(title_list)):
                l = tr()
                l += td(str(i+1))
                l += td(img(src=link_list[i]))
                l += td(title_list[i] + ' by ' + author_list[i] + ' (' + str(year_list[i]) +', '+ category_list[i] + ')')
                l += td(a('Link', href = preview_list[i]))
                i+=1

    with open('gallery.html', 'w') as f:
        f.write(doc.render())
    f.close()


def display_in_webpage():

    app = Flask(__name__, template_folder='./')

    @app.route("/")
    def hello():
        return render_template('gallery.html')
    app.run()


def main():
    # gather user input
    greeting = 'Welcome to the Book Recommedation System :)'
    print(greeting)
    user_short = input('Do you prefer shorter books? (y/n) ')
    user_category = input('Do you prefer books about cooking/biography? (yn/ny) ')
    user_recent = input('Do you prefer books published within a decade? (y/n) ')
    DISPLAY_OPTION = input('Display option? (web/txt) ')
    DISPLAY_TREE = input('Display tree? (y/n) ')


    # train decision tree
    train_X, train_y, train_info = getLibrary(user_short, user_category, user_recent)
    clf = train_decision_tree(train_X, train_y)
    
    #prediction
    test_X, test_y, test_info = getLibrary(user_short, user_category, user_recent, mode='test')
    preds_info = predict_decision_tree(clf, test_X, test_info)
    

    if DISPLAY_OPTION == 'web':
        # write html
        title_list = []
        author_list = []
        category_list = []
        link_list = []
        year_list = []
        preview_list = []
        for pred in preds_info:
            title_list.append(pred[0])
            author_list.append(pred[1])
            category_list.append(pred[2])
            link_list.append(pred[3])
            year_list.append(pred[4])
            preview_list.append(pred[5])

        write_html(title_list, author_list, category_list, link_list, year_list, preview_list, DISPLAY_TREE)
        display_in_webpage()
    else:
        print('')
        for pred in preds_info:
            print(pred[0] + ' by ' + pred[1] + ' (' +  pred[2] + ', ' + str(pred [4]) + ')')


if __name__ == '__main__':
    main()