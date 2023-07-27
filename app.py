import streamlit as st
import pickle
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

def loadData(file):
    

    file_name = '{}.pkl'.format(file)

    try:
        with open(file_name, 'rb') as file:
            loaded_data = pickle.load(file)
        print("Data successfully loaded.")
        return loaded_data
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
    except Exception as e:
        print(f"An error occurred while loading data: {e}")


            
        


def recommend(book_name , books , model , pt):
      bookId = np.where(pt.index == str(book_name))[0]
    #   print(bookId)
      distances , suggestions = model.kneighbors(pt.iloc[bookId].values.reshape(1,-1), n_neighbors=6)
      boks = []
      for i in range(len(suggestions[0])):
          if i != 0:
          # print(pt.index[suggestions[0][i]])
           boks.append(pt.index[suggestions[0][i]])

      tempdf = books[books['title'].isin(boks)]
      return tempdf

def main(books , model , pt):
    st.title("Book Recommender")

    # User input for book title (scroll down option)
    book_title = st.selectbox("Select a book title:", pt.index)

    if st.button("Recommend"):
        test = recommend(book_title,books ,model , pt)
        recommended_books = test.drop_duplicates('title')
        st.subheader("Recommended Books:")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
             st.image(recommended_books.iloc[0].url)
             st.text(recommended_books.iloc[0 ].title)
             st.text('by ' + recommended_books.iloc[0].author)
        with col2:
             st.image(recommended_books.iloc[1].url)
             st.text(recommended_books.iloc[1 ].title)
             st.text('by ' + recommended_books.iloc[1].author)
        with col3:
             st.image(recommended_books.iloc[2].url)
             st.text(recommended_books.iloc[2].title)
             st.text('by ' + recommended_books.iloc[2].author)
        with col4:
             st.image(recommended_books.iloc[3].url)
             st.text(recommended_books.iloc[3].title)
             st.text('by ' + recommended_books.iloc[3].author)
        with col5:
             st.image(recommended_books.iloc[4].url)
             st.text(recommended_books.iloc[4].title)
             st.text('by ' + recommended_books.iloc[4].author)
            


if __name__ == '__main__':
    books = loadData('books')
    model = loadData('model')
    pt = pd.read_pickle("pt.pkl")
   
    main(books , model , pt)
        
    