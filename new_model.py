from data_utils import DataUtils
import numpy as np
from keras.engine import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Layer, RepeatVector, Masking, Concatenate, Add, Reshape, Activation, Dropout, LSTM, TimeDistributed, Bidirectional, Embedding, Input
from keras.utils import plot_model
import keras.backend as K

class BiLSTM(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.f_lstm = LSTM(output_dim, dropout=0.35, recurrent_dropout=0.1, return_sequences=True)
        self.b_lstm = LSTM(output_dim, dropout=0.35, recurrent_dropout=0.1, return_sequences=True)
        self.bilstm = Concatenate()
        super(BiLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BiLSTM, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        forward_output = self.f_lstm(x[0])
        backward_output = self.b_lstm(x[1])
        bilstm_output = self.bilstm([forward_output, backward_output])
        return bilstm_output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim*2)

class DependencyParser(object):
    def __init__(self):
        pass

    def __create_xy(self, dependency_tree, embedding_file, data_size, look_back, test=False):
        sentences, words, tags = DataUtils.parse_dependency_tree(dependency_tree)
        word_vectors = DataUtils.create_onehot_vectors(words)
        #word_int = DataUtils.create_int_dict(words)
        word_emb = DataUtils.load_embeddings(embedding_file)
        tag_int = DataUtils.create_int_dict(tags)

        data_size = int(len(sentences)*min(data_size, 1))

        if test:
            sentences.reverse()

        if look_back == 0:
            for sentence in sentences[:data_size]:
                look_back = max(look_back, len(sentence))

        self.look_back = look_back
        self.distinct_words = len(words)
        self.distinct_tags = len(tags)

        word_input_forward = []
        word_input_backward = []
        word_head_forward = []
        word_head_backward = []

        tag_input_forward = []
        tag_input_backward = []
        tag_head_forward = []
        tag_head_backward = []

        probability = []

        progress = 0

        for sentence in sentences[:data_size]:
            parts = [sentence[i:i+look_back] for i in range(0,len(sentence),look_back)]
            for part in parts:
                word_temp = np.zeros((2,look_back,300))
                tag_temp = np.zeros((2,look_back,),dtype="int32")
                prob_temp = np.zeros((look_back,),dtype="float32")

                for idx in range(len(part)):
                    word = part[idx]["word"]
                    word_temp[0][look_back-len(part)+idx] = word_emb[word] if word in word_emb else word_emb["UNK"]
                    word_temp[1][look_back-idx-1] = word_emb[word] if word in word_emb else word_emb["UNK"]

                    tag = part[idx]["tag"]
                    tag_temp[0][look_back-len(part)+idx] = tag_int[tag]
                    tag_temp[1][look_back-idx-1] = tag_int[tag]

                word_instance = np.zeros((len(part),2,look_back,300))
                tag_instance = np.zeros((len(part),2,look_back,),dtype="int32")

                head_instance = np.zeros((look_back,1), dtype="float32")

                for idx in range(len(part)):
                    word_instance[idx][0][look_back-idx-1:] = word_temp[0][look_back-len(part):look_back-len(part)+idx+1]
                    word_instance[idx][1][look_back-len(part)+idx:] = word_temp[1][look_back-len(part):look_back-idx]

                    tag_instance[idx][0][look_back-idx-1:] = tag_temp[0][look_back-len(part):look_back-len(part)+idx+1]
                    tag_instance[idx][1][look_back-len(part)+idx:] = tag_temp[1][look_back-len(part):look_back-idx]

                for idx in range(len(part)):
                    word_input = np.zeros((2,2,look_back,300))
                    tag_input = np.zeros((2,2,look_back,),dtype="int32")
                    prob_temp = 0.0

                    for jdx in range(len(part)):
                        if idx != jdx:
                            if part[idx]["head"] == part[jdx]["word"]:
                                prob_temp = 1.0
                            word_input[0] = word_instance[idx]
                            tag_input[0] = tag_instance[idx]
                            word_input[1] = word_instance[jdx]
                            tag_input[1] = tag_instance[jdx]

                    if len(word_input_forward) == 0:
                        word_input_forward = [word_input[0][0]]
                        word_input_backward = [word_input[0][1]]
                        word_head_forward = [word_instance[1][0]]
                        word_head_backward = [word_instance[1][1]]

                        tag_input_forward = [tag_input[0][0]]
                        tag_input_backward = [tag_input[0][1]]
                        tag_head_forward = [tag_input[1][0]]
                        tag_head_backward = [tag_input[1][1]]

                        probability = [prob_temp]
                    else:
                        word_input_forward = np.append(word_input_forward,[word_input[0][0]], axis=0)
                        word_input_backward = np.append(word_input_backward,[word_input[0][1]], axis=0)
                        word_head_forward = np.append(word_head_forward,[word_instance[1][0]], axis=0)
                        word_head_backward = np.append(word_head_backward,[word_instance[1][1]], axis=0)

                        tag_input_forward = np.append(tag_input_forward,[tag_input[0][0]], axis=0)
                        tag_input_backward = np.append(tag_input_backward,[tag_input[0][1]], axis=0)
                        tag_head_forward = np.append(tag_head_forward,[tag_input[1][0]], axis=0)
                        tag_head_backward = np.append(tag_head_backward,[tag_input[1][1]], axis=0)

                        probability = np.append(probability, [prob_temp], axis=0)

            DataUtils.update_message(str(progress)+"/"+str(data_size))
            progress += 1

        word_data = [(word_input_forward, word_input_backward), (word_head_forward, word_head_backward)]
        tag_data = [(tag_input_forward, tag_input_backward), (tag_head_forward, tag_head_backward)]

        return word_data, tag_data, probability

    def create_xy_test(self, dependency_tree, embedding_file, data_size=1, look_back=0, mode="create", load=None):
        DataUtils.message("Prepearing Test Data...", new=True)

        if mode == "create" or mode == "save":
            word_test, head_test, tag_test = self.__create_xy(dependency_tree, embedding_file, data_size, look_back, test=True)

        if mode == "save":
            DataUtils.save_array(DataUtils.get_filename("DP_W","TEST"+"_"+str(look_back)), word_test)
            DataUtils.save_array(DataUtils.get_filename("DP_H","TEST"+"_"+str(look_back)), head_test)
            DataUtils.save_array(DataUtils.get_filename("DP_T","TEST"+"_"+str(look_back)), tag_test)

        if mode == "load" and load is not None:
            word_test = DataUtils.load_array(load[0])
            head_test = DataUtils.load_array(load[1])
            tag_test = DataUtils.load_array(load[2])

        self.word_test = np.array(word_test)
        self.head_test = np.array(head_test)
        self.tag_test = np.array(tag_test)

    def create_xy_train(self, dependency_tree, embedding_file, data_size=1, look_back=0, mode="create", load=None):
        DataUtils.message("Prepearing Training Data...", new=True)

        if mode == "create" or mode == "save":
            word_train, tag_train, probability_train = self.__create_xy(dependency_tree, embedding_file, data_size, look_back, test=False)

        self.word_train = word_train
        self.tag_train = tag_train
        self.probability_train = probability_train

    def save(self, note=""):
        DataUtils.message("Saving Model...", new=True)
        directory = "weights/"

        DataUtils.create_dir(directory)

        file = DataUtils.get_filename("DP", note)+".h5"

        self.model.save(directory+file)

    def load(self, file):
        DataUtils.message("Loading Model...", new=True)
        self.model = load_model(file)

    def plot(self, note=""):
        DataUtils.message("Ploting Model...", new=True)
        directory = "plot/"

        DataUtils.create_dir(directory)

        file = DataUtils.get_filename("DP", note)+".png"

        plot_model(self.model, to_file=directory+file, show_shapes=True, show_layer_names=False)

    def create(self):
        DataUtils.message("Creating The Model...", new=True)
        word_input_forward = Input(shape=(self.look_back,300))
        word_input_backward = Input(shape=(self.look_back,300))

        tag_input_forward = Input(shape=(self.look_back,))
        tag_input_backward = Input(shape=(self.look_back,))

        tag_emb = Embedding(self.distinct_tags, 30, input_length=self.look_back, trainable=True)
        tag_input_forward_output = tag_emb(tag_input_forward)
        tag_input_backward_output = tag_emb(tag_input_backward)

        input_forward = Concatenate()([word_input_forward, tag_input_forward_output])
        input_backward = Concatenate()([word_input_backward, tag_input_backward_output])

        word_head_forward = Input(shape=(self.look_back,300))
        word_head_backward = Input(shape=(self.look_back,300))

        tag_head_forward = Input(shape=(self.look_back,))
        tag_head_backward = Input(shape=(self.look_back,))

        tag_head_forward_output = tag_emb(tag_head_forward)
        tag_head_backward_output = tag_emb(tag_head_backward)

        head_forward = Concatenate()([word_head_forward, tag_head_forward_output])
        head_backward = Concatenate()([word_head_backward, tag_head_backward_output])

        bilstm = BiLSTM(300)

        bilstm_input = bilstm([input_forward,input_backward])
        dense_input = Dense(600, activation="linear")(bilstm_input)

        bilstm_head = bilstm([head_forward,head_backward])
        dense_head = Dense(600, activation="linear")(bilstm_head)

        sum_dense = Add()([dense_input,dense_head])

        dense_tanh = Dense(600, activation="tanh")(sum_dense)
        output = Dense(1, activation="softmax")(dense_tanh)

        model = Model(inputs=[word_input_forward, word_input_backward, tag_input_forward, tag_input_backward, word_head_forward, word_head_backward, tag_head_forward, tag_head_backward], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
        self.model = model

    def train(self, epochs, batch_size=32):
        DataUtils.message("Training...", new=True)
        a = self.model.fit([self.word_train[0][0], self.word_train[0][1], self.tag_train[0][0], self.tag_train[0][1], self.word_train[1][0], self.word_train[1][1], self.tag_train[1][0], self.tag_train[1][1]], self.probability_train, epochs=epochs, batch_size=batch_size)
        print(a.history)
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
    epochs = 0
    look_back = 100 #0 means the largest window

    model = DependencyParser()
    model.create_xy_train(train_file, embedding_file, 0.001, look_back = look_back)
    model.create()
    model.summary()
    model.train(epochs)
