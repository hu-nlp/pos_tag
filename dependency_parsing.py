from data_utils import DataUtils
import numpy as np
from keras.engine import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, LSTM, TimeDistributed, Bidirectional, Concatenate, Embedding
from keras.utils import plot_model

class SupervisedLSTM(object):
    def __init__(self):
        self.INPUT_SHAPE = [(None,),(None,)]
        self.OUTPUT_SHAPE = (None,)

    def __create_xy(self, dependency_tree, embedding_file, data_size):
        sentences, words, tags = DataUtils.parse_dependency_tree(dependency_tree)
        word_vectors = DataUtils.create_onehot_vectors(words)
        tag_vectors = DataUtils.create_onehot_vectors(tags)
        word_emb = DataUtils.load_embeddings(embedding_file)

        look_back = 0
        for sentence in sentences:
            look_back = max(look_back, len(sentence))

        data_size = int(len(sentences)*min(data_size, 1))
        word_data = []
        head_data = []
        tag_data = []

        progress = 0
        for sentence in sentences[:data_size]:
            word_timestep = np.zeros((look_back,300))
            head_timestep = np.zeros((look_back,len(words)))
            tag_timestep = np.zeros((look_back,len(tags)))

            for idx in range(len(sentence)):
                word = sentence[idx]["word"]
                if word != "ROOT":
                    word_timestep[idx] = word_emb[word] if word in word_emb else word_emb["UNK"]

                    head = sentence[idx]["head"]
                    head_timestep[idx] = word_vectors[head]

                    tag = sentence[idx]["tag"]
                    tag_timestep[idx] = tag_vectors[tag]

            if len(word_data) == 0:
                word_data = [word_timestep]
                head_data = [head_timestep]
                tag_data = [tag_timestep]
            else:
                word_data = np.append(word_data, [word_timestep], axis=0)
                head_data = np.append(head_data, [head_timestep], axis=0)
                tag_data = np.append(tag_data, [tag_timestep], axis=0)

            if progress%int(max(data_size,100)/100) == 0:
                DataUtils.update_message(str(int(progress/max(data_size,100)*100)))

            progress += 1

        word_data = np.array(word_data)
        head_data = np.array(head_data)
        tag_data = np.array(tag_data)

        return word_data, head_data, tag_data

    def create_xy_test(self, dependency_tree, embedding_file, data_size=1):
        DataUtils.message("Prepearing Test Data...", new=True)

        word_test, head_test, tag_test = self.__create_xy(dependency_tree, embedding_file, data_size)

        self.word_test = np.array(word_test)
        self.head_test = np.array(head_test)
        self.tag_test = np.array(tag_test)

    def create_xy_train(self, dependency_tree, embedding_file, data_size=1):
        DataUtils.message("Prepearing Training Data...", new=True)

        word_train, head_train, tag_train = self.__create_xy(dependency_tree, embedding_file, data_size)

        self.word_train = np.array(word_train)
        self.head_train = np.array(head_train)
        self.tag_train = np.array(tag_train)

        self.INPUT_SHAPE = [word_train.shape, tag_train.shape]
        self.OUTPUT_SHAPE = head_train.shape

    def save(self, note=""):
        DataUtils.message("Saving Model...", new=True)
        self.model.save(DataUtils.get_filename("SLSTM", note)+".h5")

    def load(self, file):
        DataUtils.message("Loading Model...", new=True)
        self.model = load_model(file)

    def plot(self, note=""):
        DataUtils.message("Ploting Model...", new=True)
        plot_model(self.model, to_file=DataUtils.get_filename("SLSTM", note)+".png", show_shapes=True, show_layer_names=False)

    def create(self):
        DataUtils.message("Creating The Model...", new=True)
        model = Sequential()
        #TODO: Create an embedding layer for tag_train, merge with word_train and take output to bilstm, train dense to head_train
        self.model = model

    def train(self, epochs, batch_size=32):
        DataUtils.message("Training...", new=True)
        self.model.fit([self.word_train,self.tag_train], self.head_train, epochs=epochs, batch_size=batch_size)

    def validate(self, batch_size=16):
        DataUtils.message("Validation...")
        return self.model.evaluate([self.word_test,self.tag_test], self.head_test, batch_size=16)

    def predict(self, x):
        return self.model.predict(x)

    def summary(self):
        self.model.summary()

if __name__ == "__main__":
    train_file = "data/penn-treebank.conllx"
    embedding_file = "embeddings/GoogleNews-vectors-negative300-SLIM.bin"
    epochs = 30

    model = SupervisedLSTM()
    model.create_xy_train(train_file, embedding_file, 0.001)
    model.create_xy_test(train_file, embedding_file, 0.001)
    model.train(30)

    DataUtils.message(model.validate())
