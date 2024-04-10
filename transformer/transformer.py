import tensorflow as tf
from tensorflow import keras
from keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    
    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
import random
import pickle
import numpy as np

limit_lens = [512, 1024]
epochs = [30,50,100]
embed_dims = [8, 16]
num_heads = [8, 16]
ff_dims = [8, 16]
batch_sizes = [16, 32, 64]

for limit_len_index in range(len(limit_lens)):
    pickle_data = pickle.load(open(f"C:\\Zhijie\\fyp\\fyp\\transformer\\input_diff_limit_len_{limit_lens[limit_len_index]}.pkl", 'rb'))
    for epoch_num in epochs:
        for embed_dim_num in embed_dims:
            for num_heads_num in num_heads:
                for ff_dim_num in ff_dims:
                    for batch_size in batch_sizes:
                        limit_len = limit_lens[limit_len_index]

                        random.shuffle(pickle_data)
                        split_ratio = 0.8
                        split_index = int(len(pickle_data) * split_ratio)

                        train_data = pickle_data[:split_index]
                        test_data = pickle_data[split_index:]

                        x_train = list()
                        y_train = list()
                        x_test = list()
                        y_test = list()
                        for elem in train_data:
                            x_train.append(elem[0])
                            y_train.append(elem[1])
                        for elem in test_data:
                            x_test.append(elem[0])
                            y_test.append(elem[1])

                        vocab_size = 30001  
                        maxlen = limit_lens[limit_len_index]


                        embed_dim = embed_dim_num  

                        num_head = num_heads_num  

                        ff_dim = ff_dim_num  

                        inputs = layers.Input(shape=(maxlen,))

                        embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)

                        x = embedding_layer(inputs)

                        transformer_block = TransformerBlock(embed_dim, num_head, ff_dim)

                        x = transformer_block(x)

                        x = layers.GlobalAveragePooling1D()(x)

                        x = layers.Dropout(0.1)(x)

                        x = layers.Dense(20, activation="relu")(x)

                        x = layers.Dropout(0.1)(x)

                        outputs = layers.Dense(2, activation="softmax")(x)

                        model = keras.Model(inputs=inputs, outputs=outputs)

                        model.compile(
                            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
                        )
                        history = model.fit(
                            x_train, y_train, batch_size, epochs=epoch_num, validation_data=(x_test, y_test)
                        )


                        def transformer_predict(x_test):
                            prediction = model.predict(x_test)
                            y_pred = [np.argmax(pred) for pred in prediction]
                            return y_pred
                        y_pred = transformer_predict(x_test)


                        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
                        accuracy = accuracy_score(y_test, y_pred)

                        precision = precision_score(y_test, y_pred)

                        recall = recall_score(y_test, y_pred)

                        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
                        roc_auc = auc(fpr, tpr)

                        print("Accuracy:", accuracy)
                        print("Precision:", precision)
                        print("Recall:", recall)
                        print("AUC:", roc_auc)
                        with open("C:\\Zhijie\\fyp\\fyp\\transformer\\transformer_finetune_result.txt", "a") as f:
                            f.write(f"AUC: {roc_auc}, limit_len:{limit_len}, epoch_num:{epoch_num}, embed_dim_num:{embed_dim_num}, \
                                    num_heads_num:{num_heads_num}, ff_dim_num:{ff_dim_num}, batch_size:{batch_size}\n")