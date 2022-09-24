import os
import requests
def get_fables():
    """
    The function `get_fables()` downloads the text of Aesop's Fables from Project Gutenberg and saves
    it to a file in the `data/books` directory
    """
    book_path = os.path.join('data','books')
    resp = requests.get("https://www.gutenberg.org/files/49010/49010-0.txt")
    if not os.path.exists(book_path):
            os.makedirs(book_path)
    with open(os.path.join(book_path,'AesopsFables.txt'), 'wb') as txt_file:
        txt_file.write(resp.content)

def counter(file_name : str) -> None:
    """
    It counts the number of words, lines, characters, and sentences in a file
    
    :param file_name: the name of the file to be read
    :type file_name: str
    """
    num_words = 0
    num_lines = 0
    num_chars = 0
    num_sentences = 0
    with open(file_name, 'r') as file:
        for line in file:
            num_lines += 1
            boolean = True
            for letter in line:
                if (letter != ' ' and boolean == True):
                    num_words += 1
                    boolean = False
                for i in letter:
                    if (i == "!" or i == "." or i == "?"): num_sentences += 1
                    if (i != " " and i != "\n"): num_chars += 1
    print("Word Count: ", num_words)
    print("Line Count: ", num_lines)
    print("Char Count: ", num_chars)
    print("Sentence Count: ", num_sentences)