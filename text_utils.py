## List of functions for text processing
## From https://towardsdatascience.com/what-people-write-about-climate-twitter-data-clustering-in-python-2fbbd2b95906
import re

from typing import List

from nltk.tokenize import TweetTokenizer
from nltk import word_tokenize

from nltk.corpus import stopwords
stop = set(stopwords.words("english"))

import spacy
nlp = spacy.load('en_core_web_sm')

def remove_stopwords(text) -> str:
    """ Remove stopwords from text """
    filtered_words = [word for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)

def expand_hashtag(tag: str):
    """ Convert #HashTag to separated words.
    '#ActOnClimate' => 'Act On Climate'
    '#climate' => 'climate' """
    res = re.findall('[A-Z]+[^A-Z]*', tag)
    return ' '.join(res) if len(res) > 0 else tag[1:]

def expand_hashtags(s: str):
    """ Convert string with hashtags.
    '#ActOnClimate now' => 'Act On Climate now' """
    res = re.findall(r'#\w+', s) 
    s_out = s
    for tag in re.findall(r'#\w+', s):
        s_out = s_out.replace(tag, expand_hashtag(tag))
    return s_out

def remove_last_hashtags(s: str):
    """ Remove all hashtags at the end of the text except #url """
    # "Change in #mind AP #News #Environment" => "Change in #mind AP"
    tokens = TweetTokenizer().tokenize(s)
    # If the URL was added, keep it
    url = "#url" if "#url" in tokens else None
    # Remove hashtags
    while len(tokens) > 0 and tokens[-1].startswith("#"):
        tokens = tokens[:-1]
    # Restore 'url' if it was added
    if url is not None:
        tokens.append(url)
    return ' '.join(tokens) 

def lemmatize(sentence: str) -> str:
    """ Convert all words in sentence to lemmatized form """
    return " ".join([token.lemma_ for token in nlp(sentence)])

def text_clean(s_text: str) -> str:
    """ Text clean """
    try:
        #output = re.sub(r"https?://\S+", "#url", s_text)  # Replace hyperlinks with '#url'
        output = re.sub(r"https?://\S+", "", s_text)  # Replace hyperlinks with '#url'
        output = re.sub(r"RT ", "", output)  # Remove RT tag
        output = re.sub(r'@\w+', '', output)  # Remove mentioned user names @... 
        output = remove_last_hashtags(output)  # Remove hashtags from the end of a string
        output = expand_hashtags(output)  # Expand hashtags to words
        output = re.sub("[^a-zA-Z]+", " ", output) # Filter
        output = re.sub(r"\s+", " ", output)  # Remove multiple spaces
        output = remove_stopwords(output)  # Remove stopwords
        output = lemmatize(output)  # Remove stopwords
        return output.lower().strip()
    except:
        return ""

def partial_clean(s_text: str) -> str:
    """ Convert tweet to a plain text sentence """
    output = re.sub(r"https?://\S+", "#url", s_text)  # Replace hyperlinks with '#url'
    output = re.sub(r"RT ", "", output)  # Remove RT tag
    output = re.sub(r'@\w+', '', output)  # Remove mentioned user names @... 
    output = remove_last_hashtags(output)  # Remove hashtags from the end of a string
    output = expand_hashtags(output)  # Expand hashtags to words
    output = re.sub(r"\s+", " ", output)  # Remove multiple spaces
    return output

def text_to_tokens(text: str) -> List[str]:
    """ Generate tokens from the sentence """
    # "this is text" => ['this', 'is' 'text']
    tokens = word_tokenize(text)  # Get tokens from text
    tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens
    return tokens

def text_len(s_text: str) -> int:
    """ Length of the text """
    return len(s_text)