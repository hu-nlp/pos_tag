from data_utils import DataUtils
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import plot_model

class UnsupervisedFF(object):
    def __init__(self):
        self.INPUT_SHAPE = (None,)
        self.OUTPUT_SHAPE = (None,)

        self.x_train = np.array([])
        self.y_train = np.array([])
        self.x_test = np.array([])
        self.y_train = np.array([])

    def __create_xy_test(self, tag_file, embedding_file, data_size, window_size, suffix):
        x_test = []
        y_test = []

        corpus = DataUtils.load_corpus(tag_file)
        tag_emb = DataUtils.create_onehot_vectors(DataUtils.extract_tag_list(corpus))
        word_emb = DataUtils.load_embeddings(embedding_file)
        if suffix is not None:
            word_emb = DataUtils.add_suffix_embeddings(word_emb, suffix[0], suffix[1])

        words, tags = DataUtils.extract_data(corpus)
        word_keys = DataUtils.normalize_cases(word_emb.keys(), words)

        data_size = min((int(len(words)*data_size), len(words)-window_size))

        for idx in range(data_size):
            word_input = word_emb[word_keys[idx]] if word_keys[idx] in word_emb else word_emb["UNK"]
            for widx in range(1, window_size):
                word_input = np.append(word_input, word_emb[word_keys[idx+widx]] if word_keys[idx+widx] in word_emb else word_emb["UNK"], axis = 0)

            tag_input = tag_emb[tags[idx+int(window_size/2)]]

            if(idx == 0):
                x_test = [word_input]
                y_test = [tag_input]
            else:
                x_test = np.append(x_test, [word_input], axis=0)
                y_test = np.append(y_test, [tag_input], axis=0)

            x_test = np.array(x_test)
            y_test = np.array(y_test)

            if idx%int(data_size/10) == 0:
                DataUtils.update_message(str(int(idx/data_size*100)))

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        return x_test, y_test

    def __create_xy_train(self, tag_file, embedding_file, data_size, window_size, threshold, suffix):
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

        data_size = min((int(len(words)*data_size), len(words)-window_size))

        for idx in range(data_size):
            word_input = word_emb[word_keys[idx]] if word_keys[idx] in word_emb else word_emb["UNK"]
            for widx in range(1, window_size):
                word_input = np.append(word_input, word_emb[word_keys[idx+widx]] if word_keys[idx+widx] in word_emb else word_emb["UNK"], axis = 0)

            tag_inputs = [tag_emb[tag] for tag in tag_dict[words[idx+int(window_size/2)]]]

            for tidx in range(len(tag_inputs)):
                tag_input = tag_inputs[tidx]
                if idx == 0 and tidx == 0:
                    x_train = [word_input]
                    y_train = [tag_input]
                else:
                    x_train = np.append(x_train, [word_input], axis=0)
                    y_train = np.append(y_train, [tag_input], axis=0)

            if idx%int(data_size/100) == 0:
                DataUtils.update_message(str(int(idx/data_size*100)))

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        return x_train, y_train

    def create_xy_test(self, tag_file, embedding_file, data_size=1, window_size=5, suffix=None, mode="create", load=None):
        DataUtils.message("Prepearing Test Data...", new=True)

        if mode == "create" or mode == "save":
            x_test, y_test = self.__create_xy_test(tag_file, embedding_file, data_size, window_size, suffix)

        if mode == "save":
            DataUtils.save_array(DataUtils.get_filename("UFF","X_TEST"+"_"+str(window_size)), x_test)
            DataUtils.save_array(DataUtils.get_filename("UFF","Y_TEST"+"_"+str(window_size)), y_test)

        if mode == "load" and load is not None:
            x_test = DataUtils.load_array(load[0])
            y_test = DataUtils.load_array(load[1])

        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)

    def create_xy_train(self, tag_file, embedding_file, data_size=1, window_size=5, threshold=0, suffix=None, mode="create", load=None):
        DataUtils.message("Prepearing Training Data...", new=True)

        if mode == "create" or mode == "save":
            x_train, y_train = self.__create_xy_train(tag_file, embedding_file, data_size, window_size, threshold, suffix)

        if mode == "save":
            DataUtils.save_array(DataUtils.get_filename("UFF","X_TRAIN"+"_"+str(window_size)), x_train)
            DataUtils.save_array(DataUtils.get_filename("UFF","Y_TRAIN"+"_"+str(window_size)), y_train)

        if mode == "load" and load is not None:
            x_train = DataUtils.load_array(load[0])
            y_train = DataUtils.load_array(load[1])

        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

        self.INPUT_SHAPE = self.x_train.shape
        self.OUTPUT_SHAPE = self.y_train.shape

    def save(self, note=""):
        DataUtils.message("Saving Model...", new=True)
        directory = "weights/"

        DataUtils.create_dir(directory)

        file = DataUtils.get_filename("UFF", note)+".h5"

        self.model.save(directory+file)

    def load(self, file):
        DataUtils.message("Loading Model...", new=True)
        self.model = load_model(file)

    def plot(self, note=""):
        DataUtils.message("Ploting Model...", new=True)
        directory = "plot/"

        DataUtils.create_dir(directory)

        file = DataUtils.get_filename("UFF", note)+".png"

        plot_model(self.model, to_file=directory+file, show_shapes=True, show_layer_names=False)

    def create(self):
        model = Sequential()
        model.add(Dropout(.2, input_shape=(self.INPUT_SHAPE[1],)))
        model.add(Dense(700, kernel_initializer="random_uniform", activation="tanh"))
        model.add(Dense(self.OUTPUT_SHAPE[1], activation="softmax"))
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.model = model

    def train(self, epochs, batch_size=16):
        DataUtils.message("Training...", new=True)
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)

    def validate(self, batch_size=16):
        DataUtils.message("Validation...")
        return self.model.evaluate(self.x_test, self.y_test, batch_size=batch_size)

    def predict(self, x):
        return self.model.predict(x)

    def summary(self, x):
        self.model.summary()

if __name__ == "__main__":
    test_file = "data/Brown_tagged_train.txt"
    train_file = "data/Brown_tagged_test.txt"
    embedding_file = "embeddings/GoogleNews-vectors-negative300-SLIM.bin"
    epochs = 10

    model = UnsupervisedFF()
    model.create_xy_train(train_file, embedding_file, 1, threshold=0)
    model.create_xy_test(test_file, embedding_file, 1)
    model.create()
    model.train(epochs)

    DataUtils.message(model.validate())
