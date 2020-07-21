def uniform(weights, optimizer, sparsity, pruning_config, extract_name_fn):
  mask_to_sparsity = {}
  for weight in weights:
    weight_pruner = pruning_config.get_pruner(weight)
    weight_name = weight.name

  # TODO
  return 

def erdos_renyi():
  return

def erdos_renyi_kernel():
  return


class Bernouilli(tf.keras.initializers.Initializer):
  """
  Mask initialization distribution.
  """

  def __init__(self, p=None):
    """
    p: probability parameter of success (i.e. 1).
    If p is None, will sample randomly from uniform distribution for sparsity.
    """
    self.p = tf.Variable(p)

  def get_config(self):
    return {'p': self.p}

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def __call__(self, shape, dytpes=tf.float32):
    probs = tf.random.uniform(shape=list(shape))
    if self.p:
      probs = probs * 0 + self.p
    # probs = tf.constant(probs, dtype=dtype)
    uniform = tf.random.uniform(shape)
    mask = tf.less(uniform, probs)
    return tf.cast(mask, tf.float32)
