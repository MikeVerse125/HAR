
class AttentionModel():
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()

    def build_model(self):
        # Define the input layers
        input_layer = Input(shape=(self.config.max_sequence_length,))

        # Embedding layer
        embedding_layer = Embedding(input_dim=self.config.vocab_size,
                                    output_dim=self.config.embedding_size,
                                    input_length=self.config.max_sequence_length)(input_layer)

        # Bi-LSTM layer
        lstm_layer = Bidirectional(LSTM(units=self.config.lstm_units,
                                        return_sequences=True))(embedding_layer)

        # Attention layer
        attention = Attention()([lstm_layer, lstm_layer])

        # Output layer
        output = Dense(units=self.config.num_classes, activation='softmax')(attention)

        # Create the model
        model = Model(inputs=input_layer, outputs=output)

        return model

    def compile_model(self):
        self.model.compile(optimizer=self.config.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, y_train, X_valid, y_valid):
        self.model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=self.config.num_epochs, batch_size=self.config.batch_size)

    def evaluate_model(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)

    def summary(self):
        self.model.summary()