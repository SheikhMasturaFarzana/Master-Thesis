## Master-Thesis
# Enhancing Term Based Document Retrieval by Word Embedding and Transformer Models

Due to space constraint I could not add the `ELIB` word embedding model and the fine-tuned transformer models. Please download and decompress all three folders into the main branch from this [link](https://drive.google.com/drive/folders/1N05EaOQaF1jSNhVZ66oNPzkFQl6G5BD6?usp=sharing) . Afterwards, download the `ft_en_cc` model using the following command and place it inside the `models` folder, 

```
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')  # English
ft = fasttext.load_model('cc.en.300.bin')
```
