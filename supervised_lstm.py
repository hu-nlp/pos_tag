from data_utils import DataUtils
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, LSTM, TimeDistributed, Bidirectional
from keras.utils import plot_model

class SupervisedLSTM(object):
    def __init__(self):
        self.INPUT_SHAPE = (None,)
        self.OUTPUT_SHAPE = (None,)

        self.x_train = np.array([])
        self.y_train = np.array([])
        self.x_test = np.array([])
        self.y_train = np.array([])

    def __create_xy(self, tag_file, embedding_file, data_size, look_back, suffix):
        x = []
        y = []

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
                x = [x_timestep]
                y = [y_timestep]
            else:
                x = np.append(x, [x_timestep], axis=0)
                y = np.append(y, [y_timestep], axis=0)

            if idx%int(data_size/(10*look_back)) == 0:
                DataUtils.update_message(str(int(idx/data_size*100)))

        return x, y

    def create_xy_test(self, tag_file, embedding_file, data_size=1, look_back=5, suffix=None, mode="create", load=None):
        DataUtils.message("Prepearing Test Data...", new=True)

        if mode == "create" or mode == "save":
            x_test, y_test = self.__create_xy(tag_file, embedding_file, data_size, look_back, suffix)

        if mode == "save":
            DataUtils.save_array(DataUtils.get_filename("SLSTM","X_TEST"+"_"+str(look_back)), x_test)
            DataUtils.save_array(DataUtils.get_filename("SLSTM","Y_TEST"+"_"+str(look_back)), y_test)

        if mode == "load" and load is not None:
            x_test = DataUtils.load_array(load[0])
            y_test = DataUtils.load_array(load[1])

        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)

    def create_xy_train(self, tag_file, embedding_file, data_size=1, look_back=5, suffix=None, mode="create", load=None):
        DataUtils.message("Prepearing Training Data...", new=True)

        if mode == "create" or mode == "save":
            x_train, y_train = self.__create_xy(tag_file, embedding_file, data_size, look_back, suffix)

        if mode == "save":
            DataUtils.save_array(DataUtils.get_filename("SLSTM","X_TRAIN"+"_"+str(look_back)), x_train)
            DataUtils.save_array(DataUtils.get_filename("SLSTM","Y_TRAIN"+"_"+str(look_back)), y_train)

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

        file = DataUtils.get_filename("SLSTM", note)+".h5"

        self.model.save(directory+file)

    def load(self, file):
        DataUtils.message("Loading Model...", new=True)
        self.model = load_model(file)

    def plot(self, note=""):
        DataUtils.message("Ploting Model...", new=True)
        directory = "plot/"

        DataUtils.create_dir(directory)

        file = DataUtils.get_filename("SLSTM", note)+".png"

        plot_model(self.model, to_file=directory+file, show_shapes=True, show_layer_names=False)

    def create(self):
        DataUtils.message("Creating The Model...", new=True)
        model = Sequential()
        model.add(Dropout(.2, input_shape=(self.INPUT_SHAPE[1], self.INPUT_SHAPE[2])))
        model.add(Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
        model.add(TimeDistributed(Dense(32, activation="tanh")))
        model.add(TimeDistributed(Dense(self.OUTPUT_SHAPE[2], activation="softmax")))
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

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

    model = SupervisedLSTM()
    model.create_xy_train(train_file, embedding_file, 1)
    model.create_xy_test(test_file, embedding_file, 1)
    model.create()
    model.train(epochs)

    DataUtils.message(model.validate())
