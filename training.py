from keras.models import Model
from keras.layers import Layer,Input
from keras.optimizers import Adam
import keras.backend as K
from Inception_RestnetV1 import InceptionResNetV1
from data_processing import get_data


class TripletLossLayer(Layer):
    def __init__(self, margin, **kwargs):
        self.margin = margin
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs

        anchor = anchor / K.sqrt(K.maximum(K.sum(K.square(anchor), axis=1, keepdims=True), 1e-10))
        positive = positive / K.sqrt(K.maximum(K.sum(K.square(positive), axis=1, keepdims=True), 1e-10))
        negative = negative / K.sqrt(K.maximum(K.sum(K.square(negative), axis=1, keepdims=True), 1e-10))

        p_dist = K.sqrt(K.sum(K.square(anchor - positive), axis=1))
        n_dist = K.sqrt(K.sum(K.square(anchor - negative), axis=1))

        return K.sum(K.maximum(p_dist - n_dist + self.margin, 0))

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

def create_model_triplet(input_shape, model, margin=0.5):
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input")

    A = model(anchor_input)
    P = model(positive_input)
    N = model(negative_input)

    output = TripletLossLayer(margin=margin, name='triplet_loss')([A, P, N])

    model_triplet = Model(inputs=[anchor_input, positive_input, negative_input], outputs=output, name="Trainer_Model")

    return model_triplet

# call model InceptionResNetV1
model = InceptionResNetV1()

# call model training with triplet loss
trainer_model = create_model_triplet((160, 160, 3), model, margin = 0.5)
trainer_model.compile(optimizer = Adam(lr=0.001))
trainer_model.summary()

# get data
data_train = get_data('data')

trainer_model.fit_generator(data_train, epochs=200)

model.save('model.h5')