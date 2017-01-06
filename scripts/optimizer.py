from keras.optimizers import Adam, SGD

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1.0)
adam = Adam(clipnorm=1.0)
