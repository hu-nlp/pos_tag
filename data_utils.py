from gensim.models import KeyedVectors
import re
import numpy as np
import os
import time
import datetime
import pickle

class DataUtils(object):
    @staticmethod
    def message(text, new=False):
        if new:
            print("\n-------------------------\n")
        print(text)

    @staticmethod
    def update_message(text):
        print("Progress:", text, "%", end='\r')

    @staticmethod
    def get_filename(model, note=""):
        if note == "":
            return model + " - " + datetime.datetime.fromtimestamp(time.time()).strftime('%d%m%Y %H%M%S')
        else:
            return model + " - " + note + " - " + datetime.datetime.fromtimestamp(time.time()).strftime('%d%m%Y %H%M%S')

    @staticmethod
    def create_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def load_corpus(file):
        corpus = ""
        with open(file) as text:
            for line in text:
                corpus += " " + line.strip()

        corpus = re.sub("\n", " ", corpus)
        corpus = re.sub("\s{2,}", " ", corpus)
        return corpus.strip()

    @staticmethod
    def clean_gold_data(corpus):
        text = ""
        words = DataUtils.extract_word_data(corpus)
        for word in words:
            text += word + " "

        return text.strip()

    @staticmethod
    def load_embeddings(file, type=None):
        DataUtils.message('Loading Embeddings...')

        file = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings", file)

        if type == None:
            file_type = file.rsplit(".",1)[1] if '.' in file else None
            if file_type == "p":
                type = "pickle"
            elif file_type == "bin":
                type = "word2vec"
            else:
                type = "word2vec"

        if type == "word2vec":
            model = KeyedVectors.load_word2vec_format(file, binary=True, unicode_errors="ignore")
            words = model.index2entity
            vectors = {word : model[word] for word in words}
        elif type == "fasttext":
            model = FastText.load_fasttext_format(file)
            words = [w for w in model.wv.vocab]
            vectors = {word: model[word] for word in words}
        elif type == "pickle":
            with open(file,'rb') as fp:
                u = pickle._Unpickler(fp)
                u.encoding = 'ISO-8859-1'
                vectors = u.load()
            words = vectors.keys()

        if "UNK" not in vectors:
            unk = np.mean([vectors[word] for word in words], axis=0)
            vectors["UNK"] = unk

        return vectors

    @staticmethod
    def save_embeddings(file, vectors, type=None):
        DataUtils.message('Saving Embeddings...')
        directory = "embeddings/"

        DataUtils.create_dir(directory)

        if type == None:
            file_type = file.rsplit(".",1)[1] if '.' in file else None
            if file_type == "p":
                type = "pickle"
            elif file_type == "bin":
                type = "word2vec"
            else:
                type = "word2vec"

        if "UNK" not in vectors:
            unk = np.mean([vectors[word] for word in vectors.keys()], axis=0)
            vectors["UNK"] = unk

        if type == "word2vec":
            pass
        elif type == "pickle":
            with open(directory+file,'wb') as fp:
                pickle.dump(vectors, fp, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def save_array(file, array):
        DataUtils.message('Saving Array...')
        directory = "arrays/"

        DataUtils.create_dir(directory)

        np.save(directory+file, array)

    def load_array(file):
        DataUtils.message('Loading Array...')
        return np.load(file)

    @staticmethod
    def create_onehot_vectors(elements):
        idx = 0
        embedding = {}

        elements.sort()
        for element in elements:
            vector = np.zeros(len(elements), dtype=int)
            vector[idx] = 1
            embedding[element] = np.array(vector)
            idx += 1

        return embedding

    @staticmethod
    def create_int_dict(elements):
        int_dict = {}

        elements.sort()
        for idx in range(len(elements)):
            int_dict[elements[idx]] = idx+1

        return int_dict

    @staticmethod
    def extract_data(corpus):
        words = []
        tags = []

        raw_data = corpus.split(" ")
        for instance in raw_data:
            word, tag = instance.rsplit("/",1)
            words.append(word)
            tags.append(tag)

        return words, tags

    @staticmethod
    def extract_tag_data(corpus):
        _, tags = DataUtils.extract_data(corpus)
        return tags

    @staticmethod
    def extract_word_data(corpus):
        words, _ = DataUtils.extract_data(corpus)
        return words

    @staticmethod
    def extract_tag_list(corpus, treshold = 0):
        tag_list = []
        raw_dict = {}

        raw_data = corpus.split(" ")
        total = 0
        for instance in raw_data:
            tag = instance.rsplit("/",1)[1]
            if tag not in raw_dict:
                raw_dict[tag] = 1
            else:
                raw_dict[tag] = raw_dict[tag] + 1
            total += 1

        for tag in raw_dict.keys():
            if raw_dict[tag]/total >= treshold:
                tag_list.append(tag)

        tag_list.sort()

        return tag_list

    @staticmethod
    def get_morph_dict(file):
        morph_dict = {}
        with open(file) as text:
            for line in text:
                line = line.strip()
                index = line.split(":")[0].lower()
                data = line.split(":")[1].split("+")[0]
                if '-' in data:
                    morph_dict[index] = data.split("-")
                else:
                    morph_dict[index] = [data]
        return morph_dict

    @staticmethod
    def get_suffix_dict(file):
        suffix_dict = {}
        with open(file) as text:
            for line in text:
                line = line.strip()
                index = line.split(":")[0].lower()
                data = line.split(":")[1].split("+")[0]
                if '-' in data:
                    data = data.rsplit("-",1)[1]
                suffix_dict[index] = data

        return suffix_dict

    @staticmethod
    def add_suffix_embeddings(word_emb, morph_file, segment_file):
        morph_emb = DataUtils.load_embeddings("embeddings/morph-vectors.p","pickle")
        suffix_dict = DataUtils.get_suffix_dict("data/metu.tr")

        for word in word_emb.keys():
            suffix = suffix_dict[word] if word in suffix_dict else "UNK"
            m_emb = morph_emb[suffix] if suffix in morph_emb else morph_emb["UNK"]
            word_emb[word] = np.concatenate((word_emb[word],m_emb),axis=0)


        word_emb["UNK"] = np.mean([word_emb[word] for word in word_emb.keys()], axis=0)

        return word_emb

    @staticmethod
    def normalize_cases(indexes, data):
        FLAG = True
        for index in indexes:
            if index != "UNK" and index.isalpha() and not index.islower():
                FLAG = False

        if FLAG:
            DataUtils.message("Key Cases Normalized!")
            normalized_data = [element.lower() for element in data]
        else:
            normalized_data = data

        return normalized_data

    @staticmethod
    def extract_tag_dict(corpus, treshold = 0.1):
        tag_dict = {}
        raw_dict = {}

        raw_data = corpus.split(" ")
        for instance in raw_data:
            word, tag = instance.rsplit("/",1)

            if word not in raw_dict:
                raw_dict[word] = {}
                raw_dict[word].update({"total": 1})
            else:
                raw_dict[word]["total"] = raw_dict[word]["total"] + 1
            if tag not in raw_dict[word]:
                raw_dict[word].update({tag: 1})
            else:
                raw_dict[word][tag] = raw_dict[word][tag] + 1

        for word in raw_dict.keys():
            tag_dict[word] = []
            for tag in raw_dict[word].keys():
                if tag != "total":
                    if raw_dict[word][tag]/raw_dict[word]["total"] >= treshold:
                        tag_dict[word].append(tag)

        return tag_dict

    @staticmethod
    def save_text(text, file):
        text_file = open(file, "w")
        text_file.write(text)
        text_file.close()

    @staticmethod
    def cartesian(input, step=0, output=None):
        input = [np.asarray(x) for x in input]
        input = np.array(input)
        if len(input) == 0:
            return output

        if output is None:
            size = np.prod([len(x) for x in input])
            output = np.zeros([size, len(input)], dtype="<U10")
        else:
            size = len(output)

        output[:,step] = np.repeat(input[0],int(size/len(input[0])))
        output = DataUtils.cartesian(input[1:],step+1,output)

        return output

    @staticmethod
    def parse_dependency_tree(language):

        base_deppars_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        conll_file = os.path.join(base_deppars_dir, language + "_train.conllx")
        if language == "english":
            return DataUtils.parse_conllx(conll_file)
        elif language == "turkish":
            return DataUtils.parse_conllx_with_morpheme(conll_file)

    @staticmethod
    def parse_conllx(file):
        corpus = ""
        with open(file) as text:
            for line in text:
                corpus += line.strip() + "\n"
        corpus = corpus.strip()

        data = corpus.split("\n")
        sentence = []
        sentences = []
        words = []
        tags = []
        for x in data:
            if len(sentence) == 0:
                sentence.append({"index": 0, "word": "ROOT", "tag": "ROOT", "head": 0, "label": "ROOT"})

            if x != "":
                x = x.split("\t")
                words.append(x[1])
                tags.append(x[3])
                y = {"index": int(x[0]), "word": x[1], "tag": x[3], "head": int(x[6]), "label": x[7]}
                sentence.append(y)
            else:
                sentences.append(sentence)
                sentence = []

        for idx in range(len(sentences)):
            for jdx in range(len(sentences[idx])):
                head_index = sentences[idx][jdx]["head"]
                word = sentences[idx][head_index]["word"]
                sentences[idx][jdx]["head"] = word
        words.append("ROOT")

        return sentences, list(set(words)), list(set(tags))

    @staticmethod
    def parse_conllx_with_morpheme(file):
        corpus = ""
        with open(file) as text:
            for line in text:
                corpus += line.strip() + "\n"
        corpus = corpus.strip()

        data = corpus.split("\n")
        sentence = []
        sentences = []
        words = []
        tags = []
        for x in data:
            if len(sentence) == 0:
                sentence.append({"index": 0, "word": "ROOT", "tag": "ROOT", "head": 0, "label": "ROOT"})

            if x != "":
                x = x.split("\t")
                index = x[0]
                index = re.sub('[^0-9]', '', index)
                if x[1] != '_':
                    words.append(x[1])
                    tags.append(x[3])
                y = {"index": int(index), "word": x[1], "tag": x[3], "head": int(x[6]), "label": x[7]}
                sentence.append(y)
            else:
                for idx, entry in enumerate(sentence):
                    i = idx
                    while sentence[i]['word'] == '_':
                        i+=1
                    sentence[idx]['head'] =  sentence[i]['index']

                mod_sentence = []
                for entry in sentence:
                    if entry['word'] != '_':
                        if sentence[entry['head']]['word'] == '_':
                            entry['head'] = sentence[entry['head']]['head']
                        mod_sentence.append(entry)

                sentences.append(mod_sentence)
                sentence = []

        for idx in range(len(sentences)):
            for jdx in range(len(sentences[idx])):
                head_index = sentences[idx][jdx]["head"]
                for entry in sentences[idx]:
                    if entry["index"] == head_index:
                        sentences[idx][jdx]["head"] = entry["word"]

        words.append("ROOT")

        return sentences, list(set(words)), list(set(tags))
