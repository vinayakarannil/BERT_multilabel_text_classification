import tensorflow as tf
from loss_util import focal_loss


class MultiLabelClassifier:

    def __init__(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(768,)))
        # model.add(hub_layer)
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        # model.add(tf.keras.layers.Dense(32, activation='relu'))
        # model.add(tf.keras.layers.Dense(32, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(rate=0.8))
        self.model.add(tf.keras.layers.Dense(6, activation='sigmoid'))
        self.model.summary()

    def train(self, x_train, y_train, x_test, y_test, optimizer='adam', loss=[focal_loss(alpha=.10, gamma=2)],
              metrics=[tf.keras.metrics.AUC()], batch_size=128, epochs=100, ):

        if metrics == 'auc':
            metrics = [tf.keras.metrics.AUC()]
        self.model.compile(optimizer, loss, metrics)

        history = self.model.fit(x_train, y_train, batch_size, epochs, validation_data=(x_test, y_test))
        results = self.model.evaluate(x_test, y_test, batch_size)
        for name, value in zip(self.model.metrics_names, results):
            print("%s: %.3f" % (name, value))
        return history, results

    def test_sample(self, test_data, actual):
        print("actual ground truth={}, predicted={}".format(actual, self.model.predict(test_data)))
