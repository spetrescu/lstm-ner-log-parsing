import sys

from src.lstm_ner import *


def main():
    train_examples_path = "data/richparsing2022/train.json"
    test_examples_path = "data/richparsing2022/test.json"
    valid_examples_path = "data/richparsing2022/valid.json"

    labels = get_labels_in_data(
        filepaths=[train_examples_path, test_examples_path, valid_examples_path]
    )

    train_examples = prepare_raw_data_to_expected_format(train_examples_path)
    test_examples = prepare_raw_data_to_expected_format(test_examples_path)
    valid_examples = prepare_raw_data_to_expected_format(valid_examples_path)

    all_text = " ".join(
        [" ".join(x[0]) for x in train_examples + valid_examples + test_examples]
    )
    vocab = sorted(set(all_text))

    train_formatted = transform_raw_data_to_lstm_input(
        data=train_examples, vocab=vocab, labels=labels
    )
    test_formatted = transform_raw_data_to_lstm_input(
        data=test_examples, vocab=vocab, labels=labels
    )
    valid_formatted = transform_raw_data_to_lstm_input(
        data=valid_examples, vocab=vocab, labels=labels
    )

    lstm_ner = LSTM_NER(train_formatted, valid_formatted, test_formatted)

    lstm_ner.gen_train_series()
    lstm_ner.gen_test_series()
    lstm_ner.gen_valid_series()

    lstm_ner.create_dataset_objects_train()
    lstm_ner.create_dataset_objects_valid()
    lstm_ner.create_dataset_objects_test()

    lstm_ner.create_padded_series_train()
    lstm_ner.create_padded_series_valid()
    lstm_ner.create_padded_series_test()

    lstm_ner.print_example_batches()

    vocab_size = lstm_ner.get_vocab_size(vocab)
    label_size = lstm_ner.get_vocab_size(labels)
    embedding_dim = 256
    rnn_units = 1024
    batch_size = 16
    
    model = lstm_ner.build_model(
        vocab_size=vocab_size+1,
        label_size=label_size+1,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=batch_size)
    
    print(model.summary())

    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    model.compile(optimizer='adam', loss=loss,metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    EPOCHS=5
    history = model.fit(lstm_ner.ds_series_batch, epochs=EPOCHS, validation_data=lstm_ner.ds_series_batch_valid,callbacks=[checkpoint_callback])

    from sklearn.metrics import classification_report, confusion_matrix

    preds = np.array([])
    y_trues= np.array([])

    for input_example_batch, target_example_batch in lstm_ner.ds_series_batch_test:

        pred=model.predict(input_example_batch, batch_size=16)
        pred_max=tf.argmax(tf.nn.softmax(pred),2).numpy().flatten()
        y_true=target_example_batch.numpy().flatten()

        preds=np.concatenate([preds,pred_max])
        y_trues=np.concatenate([y_trues,y_true])

    remove_padding = [(p,y) for p,y in zip(preds,y_trues) if y!=0]

    r_p = [x[0] for x in remove_padding]
    r_t = [x[1] for x in remove_padding]

    from collections import Counter

    print("\n\nr_p", Counter(r_p))
    print("\n\nr_t", Counter(r_t))

    print(confusion_matrix(r_p,r_t))
    print(classification_report(r_p,r_t))

if __name__ == "__main__":
    main()
