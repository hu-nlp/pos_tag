from data_utils import DataUtils
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Masking, Dense, Activation, Dropout, LSTM, TimeDistributed, Bidirectional
from keras.utils import plot_model

class UnsupervisedLSTM(object):
    def __init__(self):
        self.INPUT_SHAPE = (None,)
        self.OUTPUT_SHAPE = (None,)

        self.x_train = np.array([])
        self.y_train = np.array([])
        self.x_test = np.array([])
        self.y_train = np.array([])

    def __create_xy_test(self, tag_file, embedding_file, data_size=1, look_back=5, suffix=None):
        x_test = []
        y_test = []

        corpus = DataUtils.load_corpus(tag_file)
        tag_emb = DataUtils.create_onehot_vectors(DataUtils.extract_tag_list(corpus))
        word_emb = DataUtils.load_embeddings(embedding_file)
        if suffix is not None:
            word_emb = DataUtils.add_suffix_embeddings(word_emb, suffix[0], suffix[1])

        words, tags = DataUtils.extract_data(corpus)
        word_keys = DataUtils.normalize_cases(word_emb.keys(), words)

        data_size = int(len(words)*min(data_size, 1)) - int(len(words)*min(data_size, 1))%look_back

        for idx in np.arange(0,data_size,look_back):
            x_timestep = []
            y_timestep = []

            for jdx in range(look_back):
                word_input = word_emb[word_keys[idx+jdx]] if word_keys[idx+jdx] in word_emb else word_emb["UNK"]
                tag_input = tag_emb[tags[idx+jdx]]

                if(jdx == 0):
                    x_timestep = [word_input]
                    y_timestep = [tag_input]
                else:
                    x_timestep = np.append(x_timestep, [word_input], axis=0)
                    y_timestep = np.append(y_timestep, [tag_input], axis=0)

                x_timestep = np.array(x_timestep)
                y_timestep = np.array(y_timestep)

            if(idx == 0):
                x_test = [x_timestep]
                y_test = [y_timestep]
            else:
                x_test = np.append(x_test, [x_timestep], axis=0)
                y_test = np.append(y_test, [y_timestep], axis=0)

            if idx%int(data_size/(10*look_back)) == 0:
                DataUtils.update_message(str(int(idx/data_size*100)))

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        return x_test, y_test

    def __create_xy_train(self, tag_file, embedding_file, data_size=1, look_back=5, threshold=0, suffix=None):
        x_train = []
        y_train = []

        corpus = DataUtils.load_corpus(tag_file)
        tag_emb = DataUtils.create_onehot_vectors(DataUtils.extract_tag_list(corpus))
        word_emb = DataUtils.load_embeddings(embedding_file)
        if suffix is not None:
            word_emb = DataUtils.add_suffix_embeddings(word_emb, suffix[0], suffix[1])

        words = DataUtils.extract_word_data(corpus)
        word_keys = DataUtils.normalize_cases(word_emb.keys(), words)
        tag_dict = DataUtils.extract_tag_dict(corpus, threshold)

        data_size = int(len(words)*min(data_size, 1)) - int(len(words)*min(data_size, 1))%look_back
        data_size = 53750

        for idx in np.arange(0,data_size,look_back):
            dict_tag_inputs = [tag_dict[words[idx]]]

            word_inputs = [word_emb[word_keys[idx]]] if word_keys[idx] in word_emb else [word_emb["UNK"]]
            for widx in range(1,look_back):
                word_inputs = np.append(word_inputs, [word_emb[word_keys[idx+widx]]] if word_keys[idx+widx] in word_emb else [word_emb["UNK"]], axis = 0)
                dict_tag_inputs.append(tag_dict[words[idx+widx]])

            dict_tag_inputs = DataUtils.cartesian(np.array(dict_tag_inputs))
            for jdx in range(len(dict_tag_inputs)):
                tag_inputs = [tag_emb[tag] for tag in dict_tag_inputs[jdx]]
                if idx == 0 and jdx == 0:
                    x_train = [word_inputs]
                    y_train = [tag_inputs]
                else:
                    x_train = np.append(x_train, [word_inputs], axis=0)
                    y_train = np.append(y_train, [tag_inputs], axis=0)

            if idx%int(data_size/(10*look_back)) == 0:
                DataUtils.update_message(str(int(idx/data_size*100)))

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        return x_train, y_train

    def create_xy_test(self, tag_file, embedding_file, data_size=1, look_back=5, suffix=None, mode="create", load=None):
        DataUtils.message("Prepearing Test Data...", new=True)

        if mode == "create" or mode == "save":
            x_test, y_test = self.__create_xy_test(tag_file, embedding_file, data_size, look_back, suffix)

        if mode == "save":
            DataUtils.save_array(DataUtils.get_filename("ULSTM_X","TEST"+"_"+str(look_back)), x_test)
            DataUtils.save_array(DataUtils.get_filename("ULSTM_Y","TEST"+"_"+str(look_back)), y_test)

        if mode == "load" and load is not None:
            x_test = DataUtils.load_array(load[0])
            y_test = DataUtils.load_array(load[1])

        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)

    def create_xy_train(self, tag_file, embedding_file, data_size=1, look_back=5, threshold=0, suffix=None, mode="create", load=None):
        DataUtils.message("Prepearing Training Data...", new=True)

        if mode == "create" or mode == "save":
            x_train, y_train = self.__create_xy_train(tag_file, embedding_file, data_size, look_back, threshold, suffix)

        if mode == "save":
            DataUtils.save_array(DataUtils.get_filename("ULSTM_X","TRAIN"+"_"+str(look_back)), x_train)
            DataUtils.save_array(DataUtils.get_filename("ULSTM_Y","TRAIN"+"_"+str(look_back)), y_train)

        if mode == "load" and load is not None:
            x_train = DataUtils.load_array(load[0])
            y_train = DataUtils.load_array(load[1])

        self.x_train = x_train
        self.y_train = y_train

        self.INPUT_SHAPE = x_train.shape
        self.OUTPUT_SHAPE = y_train.shape

    def save(self, note=""):
        DataUtils.message("Saving Model...", new=True)
        directory = "weights/"

        DataUtils.create_dir(directory)

        file = DataUtils.get_filename("ULSTM", note)+".h5"

        self.model.save(directory+file)

    def load(self, file):
        DataUtils.message("Loading Model...", new=True)
        self.model = load_model(file)

    def plot(self, note=""):
        DataUtils.message("Ploting Model...", new=True)
        directory = "plot/"

        DataUtils.create_dir(directory)

        file = DataUtils.get_filename("ULSTM", note)+".png"

        plot_model(self.model, to_file=directory+file, show_shapes=True, show_layer_names=False)

    def create(self):
        DataUtils.message("Creating The Model...", new=True)
        model = Sequential()
        model.add(Masking(input_shape=(self.INPUT_SHAPE[1], self.INPUT_SHAPE[2])))
        model.add(Dropout(.2))
        model.add(Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
        model.add(TimeDistributed(Dense(32, activation="tanh")))
        model.add(TimeDistributed(Dense(self.OUTPUT_SHAPE[2], activation="softmax")))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        self.model = model

    def train(self, epochs, batch_size=32):
        DataUtils.message("Training...", new=True)
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)

    def validate(self, batch_size=16):
        DataUtils.message("Validation...")
        return self.model.evaluate(self.x_test, self.y_test, batch_size=batch_size)

    def predict(self, x):
        return self.model.predict(x)

    def summary(self):
        self.model.summary()

if __name__ == "__main__":
    test_file = "data/Brown_tagged_train.txt"
    train_file = "data/Brown_tagged_test.txt"
    embedding_file = "embeddings/GoogleNews-vectors-negative300-SLIM.bin"
    epochs = 30

    model = UnsupervisedLSTM()
    model.create_xy_train(train_file, embedding_file, 1, threshold=0)
    model.create_xy_test(test_file, embedding_file, 1)
    model.create()
    model.train(epochs)

    DataUtils.message(model.validate())