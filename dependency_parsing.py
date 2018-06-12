from data_utils import DataUtils
import numpy as np
from keras.engine import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Concatenate, Reshape, Activation, Dropout, LSTM, TimeDistributed, Bidirectional, Concatenate, Embedding, Input
from keras.utils import plot_model

class SupervisedLSTM(object):
    def __init__(self):
        pass

    #creates test data
    def __create_xy(self, dependency_tree, embedding_file, data_size, look_back, test=False):
        sentences, words, tags = DataUtils.parse_dependency_tree(dependency_tree)
        word_vectors = DataUtils.create_onehot_vectors(words)
        word_emb = DataUtils.load_embeddings(embedding_file)
        tag_int = DataUtils.create_int_dict(tags)

        data_size = int(len(sentences)*min(data_size, 1))

        if test:
            sentences.reverse()

        if look_back == 0:
            for sentence in sentences[:data_size]:
                look_back = max(look_back, len(sentence))

        self.look_back = look_back
        self.num_words = len(words)
        self.num_tags = len(tags)

        word_data = []
        head_data = []
        tag_data = []

        progress = 0
        for sentence in sentences[:data_size]:
            for idx in np.arange(0,len(sentence),look_back): #Split an word > look_back into pieces i.e. 21 word to 3 pieces
                for jdx in range(min(len(sentence)-idx,look_back)): #0 to 9, 10 to 19, 20 to 20
                    word_timestep = np.zeros((look_back,300))
                    head_timestep = np.zeros((look_back,len(words)))
                    tag_timestep = np.zeros((look_back,),dtype="int32")

                    word = sentence[idx+jdx]["word"]
                    if word != "ROOT":
                        word_timestep[jdx] = word_emb[word] if word in word_emb else word_emb["UNK"]

                        head = sentence[idx+jdx]["head"]
                        head_timestep[jdx] = word_vectors[head]

                        tag = sentence[idx+jdx]["tag"]
                        tag_timestep[jdx] = tag_int[tag]

                    if len(word_data) == 0:
                        word_data = [word_timestep]
                        head_data = [head_timestep]
                        tag_data = [tag_timestep]
                    else:
                        word_data = np.append(word_data, [word_timestep], axis=0)
                        head_data = np.append(head_data, [head_timestep], axis=0)
                        tag_data = np.append(tag_data, [tag_timestep], axis=0)

            DataUtils.update_message(str(progress)+"/"+str(data_size))
            progress += 1

        word_data = np.array(word_data)
        head_data = np.array(head_data)
        tag_data = np.array(tag_data)

        return word_data, head_data, tag_data

    def create_xy_test(self, dependency_tree, embedding_file, data_size=1, look_back=0):
        DataUtils.message("Prepearing Test Data...", new=True)

        word_test, head_test, tag_test = self.__create_xy_2(dependency_tree, embedding_file, data_size, look_back, test=True)

        self.word_test = np.array(word_test)
        self.head_test = np.array(head_test)
        self.tag_test = np.array(tag_test)

    def create_xy_train(self, dependency_tree, embedding_file, data_size=1, look_back=0):
        DataUtils.message("Prepearing Training Data...", new=True)

        word_train, head_train, tag_train = self.__create_xy(dependency_tree, embedding_file, data_size, look_back, test=False)

        self.word_train = np.array(word_train)
        self.head_train = np.array(head_train)
        self.tag_train = np.array(tag_train)

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

        word_input = Input(shape=(self.look_back,300))

        tag_input = Input(shape=(self.look_back,))
        tag_emb = Embedding(self.num_tags+1, 30, input_length=self.look_back, mask_zero=True, trainable=False)(tag_input)

        concat_emb = Concatenate()([word_input, tag_emb])

        bilstm = Bidirectional(LSTM(300, dropout=0.35, recurrent_dropout=0.1, return_sequences=True))(concat_emb)
        hidden = TimeDistributed(Dense(self.num_words, activation="tanh"))(bilstm)
        output = TimeDistributed(Dense(self.num_words, activation="softmax"))(hidden)

        model = Model(inputs=[word_input, tag_input], outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

        self.model = model

    def train(self, epochs, batch_size=32):
        DataUtils.message("Training...", new=True)
        self.model.fit([self.word_train, self.tag_train], self.head_train, epochs=epochs, batch_size=batch_size)

    def validate(self, batch_size=16):
        DataUtils.message("Validation...")
        return self.model.evaluate([self.word_test,self.tag_test], self.head_test, batch_size=batch_size)

    def predict(self, x):
        return self.model.predict(x)

    def summary(self):
        self.model.summary()

if __name__ == "__main__":
    train_file = "data/penn-treebank.conllx"
    embedding_file = "embeddings/GoogleNews-vectors-negative300-SLIM.bin"
    epochs = 30
    look_back = 10 #0 means the largest window

    model = SupervisedLSTM()
    model.create_xy_train(train_file, embedding_file, 0.1, look_back = look_back)
    model.create_xy_test(train_file, embedding_file, 0.01, look_back = look_back)
    model.create()
    model.summary()
    model.train(30)

    DataUtils.message(model.validate())