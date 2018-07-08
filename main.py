import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Hyperparams
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')

tf.app.flags.DEFINE_integer('hidden_dim',    256,   'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('batch_size',    16,    'batch size')
tf.app.flags.DEFINE_integer('emb_dim',       300,   'dimension of word embeddings')
tf.app.flags.DEFINE_integer('max_enc_steps', 400,   'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100,   'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('vocab_size',    50000, 'size of vocabulary')

tf.app.flags.DEFINE_float('lr',                 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc',   0.1,  'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform initializer')
tf.app.flags.DEFINE_float('trun_norm_init_std', 1e-4, 'std of truncated normal initializer')
tf.app.flags.DEFINE_float('max_grad_norm',      2.0,  'for gradient clipping')

if __name__ == "__main__":
    tf.app.run()
