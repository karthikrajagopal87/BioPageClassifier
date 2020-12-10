import spacy
import sys
from geotext import GeoText
import os

nlp = spacy.load("en_core_web_sm")

# Function to check if the token is a noise or not 
def isNoise(token):
    is_noise = False
    if token.pos_ in noisy_pos_tags:
        is_noise = True
    elif token.is_stop == True:
        is_noise = True
    elif len(token.string) <= min_token_length:
        is_noise = True
    elif token.string.lower().strip() in stop_words:
        is_noise = True
    elif token.string.strip() in ents:
        is_noise = True
    return is_noise
    

def cleanup(token, lower = True):
    if lower:
        token = token.lower()
    return token.strip()


def spacy_modelling():
path = 'bios/'
fileList = os.listdir(path)
for i in fileList:
    file = open(os.path.join('bios/'+ i), encoding="utf8").read()
    document = nlp(file)

    # define some parameters 
    noisy_pos_tags = ["PROP"]
    min_token_length = 3

    stop_words = ('university','professor','teaching')
    ents = [e.text for e in document.ents]
    

    # top unigrams used in the reviews
    from collections import Counter	
    cleaned_list = [cleanup(word.string) for word in document if not isNoise(word)]
    counts=Counter(cleaned_list).most_common(5)
    f = open(os.path.join('bios/'+ i), "a")
    f.write("\n" + 'Key Words based on Topic Modelling done through SPACY:' + "\n")
    for key in counts:
        f.write(key[0] + "\t")
        
        
# Starting
spacy_modelling()