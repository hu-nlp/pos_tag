from pos_tag.data_utils import DataUtils
import keras
import numpy as np
from keras.engine import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Concatenate, Activation, Dropout, LSTM, TimeDistributed, Bidirectional, Embedding, Input
from keras.utils import plot_model
import matplotlib.pyplot as plt


class DependencyParser(object):
    def __init__(self):
        pass

    def __create_xy(self, dependency_tree, embedding_file, morp_file, suffix_file, data_size, look_back, language="EN",
                    test=False):
        sentences, words, tags = None, None, None
        if language == "TR":
            sentences, words, tags = DataUtils.parse_turkish_dependency_tree(dependency_tree)
        elif language == "EN":
            sentences, words, tags = DataUtils.parse_dependency_tree(dependency_tree)

        word_vectors = DataUtils.create_onehot_vectors(words)
        word_emb = DataUtils.load_embeddings(embedding_file, "fasttext")
        tag_int = DataUtils.create_int_dict(tags)
        data_size = int(len(sentences) * min(data_size, 1))
        morph_emb = DataUtils.load_embeddings(morp_file, "pickle")
        suffix_dict = DataUtils.get_suffix_dict(suffix_file)

        if test:
            sentences.reverse()

        if look_back == 0:
            for sentence in sentences[:data_size]:
                look_back = max(look_back, len(sentence))

        self.look_back = look_back
        self.distinct_words = len(words)
        self.distinct_tags = len(tags)

        word_data = []
        head_data = []
        tag_data = []
        morp_data = []

        progress = 0
        for sentence in sentences[:data_size]:
            word_timestep = np.zeros((look_back, 300))
            morp_timestep = np.zeros((look_back, 50))
            head_timestep = np.zeros((look_back, len(words)))
            tag_timestep = np.zeros((look_back,), dtype="int32")

            timestep = 0
            for element in sentence:
                word = element["word"]

                if word != "ROOT":
                    word_timestep[timestep % look_back] = word_emb[word] if word in word_emb else word_emb["UNK"]

                    head = element["head"]
                    head_timestep[timestep % look_back] = word_vectors[head]

                    tag = element["tag"]
                    tag_timestep[timestep % look_back] = tag_int[tag]

                    # morp embdeddings
                    if word in suffix_dict and suffix_dict[word] in morph_emb:
                        morp_timestep[timestep % look_back] = morph_emb[suffix_dict[word]]
                    else:
                        morp_timestep[timestep % look_back] = morph_emb['UNK']

                timestep += 1

                if timestep % look_back == 0 or timestep == len(sentence):
                    if len(word_data) == 0:
                        word_data = [word_timestep]
                        head_data = [head_timestep]
                        tag_data = [tag_timestep]
                        morp_data = [morp_timestep]
                    else:
                        word_data = np.append(word_data, [word_timestep], axis=0)
                        head_data = np.append(head_data, [head_timestep], axis=0)
                        tag_data = np.append(tag_data, [tag_timestep], axis=0)
                        morp_data = np.append(morp_data, [morp_timestep], axis=0)

                    word_timestep.fill(0)
                    head_timestep.fill(0)
                    tag_timestep.fill(0)
                    morp_timestep.fill(0)

            DataUtils.update_message(str(progress) + "/" + str(data_size))
            progress += 1

        word_data = np.array(word_data)
        head_data = np.array(head_data)
        tag_data = np.array(tag_data)
        morp_data = np.array(morp_data)

        return word_data, head_data, tag_data, morp_data

    def create_xy_test(self, dependency_tree, embedding_file, morp_file, suffix_file, data_size=1,
                       look_back=0, language="EN", mode="create", load=None):
        DataUtils.message("Prepearing Test Data...", new=True)

        if mode == "create" or mode == "save":
            word_test, head_test, tag_test, morph_test = self.__create_xy(dependency_tree, embedding_file, morp_file,
                                                                          suffix_file, data_size, look_back, language,
                                                                          test=True)

        if mode == "save":
            DataUtils.save_array(DataUtils.get_filename("DP_W", "TEST" + "_" + str(look_back)), word_test)
            DataUtils.save_array(DataUtils.get_filename("DP_H", "TEST" + "_" + str(look_back)), head_test)
            DataUtils.save_array(DataUtils.get_filename("DP_T", "TEST" + "_" + str(look_back)), tag_test)
            DataUtils.save_array(DataUtils.get_filename("DP_T", "TEST" + "_" + str(look_back)), morph_test)

        if mode == "load" and load is not None:
            word_test = DataUtils.load_array(load[0])
            head_test = DataUtils.load_array(load[1])
            tag_test = DataUtils.load_array(load[2])
            morph_test = DataUtils.load_array(load[2])

        self.word_test = np.array(word_test)
        self.head_test = np.array(head_test)
        self.tag_test = np.array(tag_test)
        self.morph_test = np.array(morph_test)

    def create_xy_train(self, dependency_tree, embedding_file, morp_file, suffix_file, data_size=1,
                        look_back=0, language="EN", mode="create", load=None):
        DataUtils.message("Prepearing Training Data...", new=True)

        if mode == "create" or mode == "save":
            word_train, head_train, tag_train, morph_train = self.__create_xy(dependency_tree, embedding_file,
                                                                              morp_file, suffix_file,
                                                                              data_size, look_back, language,
                                                                              test=False)

        if mode == "save":
            DataUtils.save_array(DataUtils.get_filename("DP_W", "TRAIN" + "_" + str(look_back)), word_train)
            DataUtils.save_array(DataUtils.get_filename("DP_H", "TRAIN" + "_" + str(look_back)), head_train)
            DataUtils.save_array(DataUtils.get_filename("DP_T", "TRAIN" + "_" + str(look_back)), tag_train)
            DataUtils.save_array(DataUtils.get_filename("DP_M", "TRAIN" + "_" + str(look_back)), morph_train)

        if mode == "load" and load is not None:
            word_train = DataUtils.load_array(load[0])
            head_train = DataUtils.load_array(load[1])
            tag_train = DataUtils.load_array(load[2])
            morph_train = DataUtils.load_array(load[3])

        self.word_train = np.array(word_train)
        self.head_train = np.array(head_train)
        self.tag_train = np.array(tag_train)
        self.morph_train = np.array(morph_train)

    def save(self, note=""):
        DataUtils.message("Saving Model...", new=True)
        directory = "weights/"

        DataUtils.create_dir(directory)

        file = DataUtils.get_filename("DP", note) + ".h5"

        self.model.save(directory + file)

    def load(self, file):
        DataUtils.message("Loading Model...", new=True)
        self.model = load_model(file)

    def plot(self, note=""):
        DataUtils.message("Ploting Model...", new=True)
        directory = "plot/"

        DataUtils.create_dir(directory)

        file = DataUtils.get_filename("DP", note) + ".png"

        plot_model(self.model, to_file=directory + file, show_shapes=True, show_layer_names=False)

    def create(self):
        DataUtils.message("Creating The Model...", new=True)

        word_input = Input(shape=(self.look_back, 300))

        morph_input = Input(shape=(self.look_back, 50))

        tag_input = Input(shape=(self.look_back,))
        tag_emb = Embedding(self.distinct_tags, 30, input_length=self.look_back, mask_zero=True)(tag_input)

        concat_emb = Concatenate()([word_input, morph_input, tag_emb])

        bilstm = Bidirectional(LSTM(300, dropout=0.35, recurrent_dropout=0.1, return_sequences=True))(concat_emb)
        hidden = TimeDistributed(Dense(800, activation="tanh"))(bilstm)
        output = TimeDistributed(Dense(self.distinct_words, activation="softmax"))(hidden)

        model = Model(inputs=[word_input, morph_input, tag_input], outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

        self.model = model

    def train(self, epochs, batch_size=32, plot=True):
        DataUtils.message("Training...", new=True)
        callbacks = [
            keras.callbacks.TensorBoard(
                log_dir='my_log_dir',
                histogram_freq=1,
                embeddings_freq=1, )]

        history = self.model.fit([self.word_train, self.morph_train, self.tag_train], self.head_train, epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=([self.word_test, self.morph_test, self.tag_test], self.head_test),
                                 callbacks=callbacks)
        if plot:
            self.plot_results(history)

    def validate(self, batch_size=16):
        DataUtils.message("Validation...")
        return self.model.evaluate([self.word_test, self.morph_test, self.tag_test], self.head_test,
                                   batch_size=batch_size)

    def predict(self, x):
        return self.model.predict(x)

    def plot_results(self, history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        # plt.figure()
        plt.savefig("plots/acc.png")
        plt.clf()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.savefig("plots/loss.png")

    def summary(self):
        self.model.summary()


if __name__ == "__main__":
    train_file = "/home/huseyin/HeadSelection/Data/METUSABANCI_treebank_v-1.conll"
    embedding_file = "/home/huseyin/HeadSelection/Data/cc.tr.300.bin"
    morph_file = "/home/huseyin/HeadSelection/Data/morph-vectors.p"
    suffix_file = "/home/huseyin/HeadSelection/Data/metu.tr"

    epochs = 5
    look_back = 10  # 0 means the largest window

    model = DependencyParser()
    model.create_xy_train(train_file, embedding_file, morph_file, suffix_file, 0.01, look_back=look_back, language="TR")
    model.create_xy_test(train_file, embedding_file, morph_file, suffix_file, 0.001, look_back=look_back, language="TR")
    model.create()
    model.train(epochs)

    # sentences, words, tags = DataUtils.parse_turkish_dependency_tree(turkish_train_file)

    # DataUtils.message(model.validate())
