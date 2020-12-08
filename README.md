## Setup
```bash
# Creating Conda environment
conda env create -f env.yml
# Activating the environment
conda activate BioPageClassifier

# Downloading NLTK Corpora
python -m nltk.downloader all
# Downloading spaCY model
python -m spacy download en_core_web_sm
# Downloading Word Embeddings Glove.6B
[ ! -d "glove.6B" ] && \
  wget http://nlp.stanford.edu/data/glove.6B.zip && \
  mkdir -p .glove.6B && \
  unzip glove.6B.zip -d .glove.6B && \
  rm glove.6B.zip
```

## Crawler
You can run crawler to collect negative data by running the following:
```bash
# Running Spider

$ scrapy runspider crawler.py \
  -a start_url=https://www.cnn.com/ \
  -a allowed_domain=cnn.com \
  -a db=/tmp/cnn_db.json \
  -a output=/tmp \
  -a exclude='.+?(show|business).+?'
```

## Classifier
Classifier model is saved in `bio-model`
### Accuracy / Loss Chart
![Accuracy Loss Chart](bio-model-history-plot.png)
### Model Architecture Chart
![Architecture](bio-model-arch-plot.png)

## Usage
Get the model
```bash
wget https://drive.google.com/file/d/1NShUBtE248LN_L1zzyGbK__4I60bkk0R/view?usp=sharing
[ ! -d "bio-model" ] && unzip bio-model.zip 
```

Load the model
```python
from utils import *

bio_identifier = BioIdentifier(model="bio-model")
# Returns False
bio_identifier.is_bio_url("https://www.cnn.com/")
# Returns True
bio_identifier.is_bio_url("https://sites.google.com/a/oakland.edu/scottcrabill/Home")
...
bio_identifier.is_bio_html_content("<html>....</html>")
```

And just to play around in python console
```python
from utils import *
processor = TextProcessor()
def get_url_text(url):
    with urllib.request.urlopen(url) as response:
        html = response.read()
    text = BeautifulSoup(html, 'html.parser').get_text(separator=" ")
    return processor.process_text(text)

model = tf.keras.models.load_model("bio-model", custom_objects={'KerasLayer': hub.KerasLayer})


def get_scores(urls):
	for u in urls:
		txt = get_url_text(u)
		print("{} - {}".format(model.predict([txt])[0][0], txt[:50]))


urls = [
    "https://cnn.com"
]

get_scores(urls)
```