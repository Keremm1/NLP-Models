from transformer import Transformer
import tensorflow as tf
import tensorflow_datasets as tfds

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1


model_name = 'ted_hrlr_translate_pt_en_converter'
tf.keras.utils.get_file(
    f'{model_name}.zip',
    f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
    cache_dir='.', cache_subdir='', extract=True
)

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

MAX_TOKENS=128
def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)      # Output is ragged.
    pt = pt[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS+1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens

    return (pt, en_inputs), en_labels


BUFFER_SIZE = 20000
BATCH_SIZE = 64

def make_batches(ds):
  return (
      ds
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(prepare_batch, tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE))

tokenizers = tf.saved_model.load(model_name)

def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
      super().__init__()

      self.d_model = d_model
      self.d_model = tf.cast(self.d_model, tf.float32)

      self.warmup_steps = warmup_steps

    def __call__(self, step):
      step = tf.cast(step, dtype=tf.float32)
      arg1 = tf.math.rsqrt(step)
      arg2 = step * (self.warmup_steps ** -1.5)

      return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)




transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    dropout_rate=dropout_rate)

transformer.summary()


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

transformer.fit(train_batches,
                epochs=20,
                validation_data=val_batches)


class Translator(tf.Module):
    def __init__(self, tokenizers, transformer):
      self.tokenizers = tokenizers
      self.transformer = transformer

    def __call__(self, sentence, max_length=MAX_TOKENS):
      # The input sentence is Portuguese, hence adding the `[START]` and `[END]` tokens.
      assert isinstance(sentence, tf.Tensor)
      if len(sentence.shape) == 0:
        sentence = sentence[tf.newaxis]

      sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

      encoder_input = sentence

      # As the output language is English, initialize the output with the
      # English `[START]` token.
      start_end = self.tokenizers.en.tokenize([''])[0]
      start = start_end[0][tf.newaxis]
      end = start_end[1][tf.newaxis]

      # `tf.TensorArray` is required here (instead of a Python list), so that the
      # dynamic-loop can be traced by `tf.function`.
      output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
      output_array = output_array.write(0, start)

      for i in tf.range(max_length):
        output = tf.transpose(output_array.stack())
        predictions = self.transformer([encoder_input, output], training=False)

        # Select the last token from the `seq_len` dimension.
        predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

        predicted_id = tf.argmax(predictions, axis=-1)

        # Concatenate the `predicted_id` to the output which is given to the
        # decoder as its input.
        output_array = output_array.write(i+1, predicted_id[0])

        if predicted_id == end:
          break

      output = tf.transpose(output_array.stack())
      # The output shape is `(1, tokens)`.
      text = tokenizers.en.detokenize(output)[0]  # Shape: `()`.

      tokens = tokenizers.en.lookup(output)[0]

      # `tf.function` prevents us from using the attention_weights that were
      # calculated on the last iteration of the loop.
      # So, recalculate them outside the loop.
      self.transformer([encoder_input, output[:,:-1]], training=False)
      attention_weights = self.transformer.decoder.last_attn_scores

      return text, tokens, attention_weights
  
translator = Translator(tokenizers, transformer)

def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')

#Testing
sentence = 'este é um problema que temos que resolver.'
ground_truth = 'this is a problem we have to solve .'

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)