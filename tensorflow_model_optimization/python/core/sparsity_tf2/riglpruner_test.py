# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the key functions in riglpruner library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import g3

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

# TODO(b/139939526): move to public API.
from tensorflow.python.keras import keras_parameterized
from tensorflow_model_optimization.python.core.sparsity_tf2 import schedule
from tensorflow_model_optimization.python.core.sparsity_tf2 import riglpruner as pruner
from tensorflow_model_optimization.python.core.sparsity_tf2 import pruning_optimizer
from tensorflow_model_optimization.python.core.sparsity_tf2 import pruning_config

dtypes = tf.dtypes
test = tf.test


def make_update_schedule(fraction, begin, end, freq):
  return schedule.ConstantSparsity(fraction, begin, end, freq)

def sample_noise(x, mu=0, sigma=1.):
  sample = tf.random.normal((), mean=mu,  stddev=sigma, dtype=tf.float64)
  return sample

def _dummy_gradient(x, dtype=tf.float32):
  try:
    base_type = x.dtype
  except:
    base_type = dtype
  # must shuffle otherwise each gradient update will be the same and 
  # connections are never regrown
  grad = tf.random.shuffle(np.linspace(1., 100., 100))
  return tf.Variable(grad, dtype=base_type)

class RiglPruningTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(RiglPruningTest, self).setUp()
    self.block_size = (1, 1)
    self.block_pooling_type = "AVG"
    self.target_sparsity = 0.5
    self.initial_drop_fraction = 0.3 # i.e. 0.3 * target sparsity per layer, annealed by schedule below
    self.constant_updater = schedule.ConstantSchedule(self.initial_drop_fraction, 0, 100, 1)
    self.skip_updater = schedule.ConstantSchedule(self.initial_drop_fraction, 0, 100, 2)
    self.updater = schedule.ConstantSchedule
    self.grad = _dummy_gradient
    self.seed = 0
    self.noise_std = 1
    self.reinit = False
    self.grow_init_zeros = 'zeros'
    self.grow_init_randn = 'random normal'
    self.grow_init_randunif = 'random uniform'

  def testGetGrowGrads(self):
    mask = tf.ones((100))
    grads = -1 * tf.linspace(1., 6., 6)
    expected_grads = tf.linspace(1., 6., 6)

    p = pruner.RiGLPruner(
      update_schedule=self.skip_updater,
      sparsity=self.target_sparsity,
      block_size=self.block_size,
      block_pooling_type=self.block_pooling_type,
      seed=self.seed,
      noise_std=self.noise_std,
      reinit=self.reinit,
      grow_init=self.grow_init_zeros
    )

    grow_grads = p._get_grow_grads(mask, grads)
    self.assertAllEqual(grow_grads, expected_grads)

  def testMaskNoChangeBeforeandAfter(self):
    weight = tf.Variable(np.linspace(1.0, 100.0, 100))
    weight_dtype = weight.dtype.base_dtype
    mask = tf.Variable(
        tf.ones(weight.get_shape(), dtype=weight_dtype),
        dtype=weight_dtype)
    sparse_vars = [(mask, weight, self.grad(weight))]

    p = pruner.RiGLPruner(
      update_schedule=self.skip_updater,
      sparsity=self.target_sparsity,
      block_size=self.block_size,
      block_pooling_type=self.block_pooling_type,
      seed=self.seed,
      noise_std=self.noise_std,
      reinit=self.reinit
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    optimizer.iterations.assign(0)
    p.create_slots(optimizer, weight)

    mask_before_update = optimizer.get_slot(weight, 'mask').read_value()
    before_sparsity = np.count_nonzero(mask_before_update) / tf.size(mask_before_update)
    self.assertAllEqual(before_sparsity, self.target_sparsity)

    next_step = optimizer.iterations.assign_add(1)
    reset_momentum, new_connections = p.update_masks(sparse_vars, next_step)
    self.assertAllEqual(reset_momentum, False)
    self.assertAllEqual(new_connections, None)

    mask_after_update = optimizer.get_slot(weight, 'mask').read_value()
    after_sparsity = np.count_nonzero(mask_after_update) / tf.size(mask_before_update)
    self.assertAllEqual(after_sparsity, self.target_sparsity)
    self.assertAllEqual(mask_after_update, mask_before_update) # no update

  @parameterized.parameters((3, 7, 2), (1, 5, 3), (0, 4, 1))
  def testDropFractionZeroNoMaskChange(self, begin_step, end_step, freq_update):
    weight = tf.Variable(np.linspace(1.0, 100.0, 100))
    weight_dtype = weight.dtype.base_dtype
    mask = tf.Variable(
        tf.ones(weight.get_shape(), dtype=weight_dtype),
        dtype=weight_dtype)
    sparse_vars = [(mask, weight, self.grad(weight))]
    drop_ratio = 0.

    p = pruner.RiGLPruner(
      update_schedule=self.updater(drop_ratio, begin_step, end_step, freq_update),
      sparsity=self.target_sparsity,
      block_size=self.block_size,
      block_pooling_type=self.block_pooling_type,
      seed=self.seed,
      noise_std=self.noise_std,
      reinit=self.reinit
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    optimizer.iterations.assign(0)
    p.create_slots(optimizer, weight)

    for _ in tf.range(end_step + 2):
      mask_before_update = optimizer.get_slot(weight, 'mask').read_value()
      before_sparsity = np.count_nonzero(mask_before_update) / tf.size(mask_before_update)
      step = optimizer.iterations
      p.update_masks(sparse_vars, step)
      mask_after_update = optimizer.get_slot(weight, 'mask').read_value()
      after_sparsity = np.count_nonzero(mask_after_update) / tf.size(mask_before_update)
      self.assertAllEqual(mask_before_update, mask_after_update)

  @parameterized.parameters(
    (0.7,), (0.1,), (0.3,), (0.5,), 
  )
  def testUpdatesAccordingtoSchedule(self, drop_ratio):
    # unlike previous tests, this checks that the masked weights
    # maintain consistent sparsity and that there is some change in the mask.
    weight = tf.Variable(np.linspace(1.0, 100.0, 100))
    weight_dtype = weight.dtype.base_dtype
    mask = tf.Variable(
        tf.ones(weight.get_shape(), dtype=weight_dtype),
        dtype=weight_dtype)
    grad = self.grad(weight)
    sparse_vars = [(mask, weight, grad)]
    sparsity_params = {
      'pruning_schedule': self.updater(drop_ratio, 1, 4, 2)
    } # dummy just for using the pruning optimizer
    sparsity_config = pruning_config.LowMagnitudePruningConfig(**sparsity_params) # dummy

    p = pruner.RiGLPruner(
      update_schedule=self.updater(drop_ratio, 1, 4, 2), # update iter 1 and 3
      sparsity=self.target_sparsity,
      block_size=self.block_size,
      block_pooling_type=self.block_pooling_type,
      seed=self.seed,
      noise_std=self.noise_std,
      reinit=self.reinit
    )

    _optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    optimizer = pruning_optimizer.PruningOptimizer(_optimizer, sparsity_config)
    optimizer.iterations.assign(0)

    p.create_slots(optimizer, weight)

    def _train(optimizer, weight):
      expected_iter1 = tf.zeros_like(weight, dtype=weight.dtype.base_dtype) * -1
      expected_iter3 = tf.zeros_like(weight, dtype=weight.dtype.base_dtype) * -1
      for i in tf.range(0, 5):
        mask_before_update = optimizer.get_slot(weight, 'mask').read_value()
        mask_before_update3 = optimizer.get_slot(weight, 'mask').read_value()
        before_sparsity = tf.math.count_nonzero(mask_before_update, dtype=tf.int32) / tf.size(mask_before_update)
        grad = p.preprocess_weights(optimizer, weight, self.grad(weight))
        weight.assign(tf.math.add(weight, sample_noise(i)))
        grad.assign(tf.math.add(grad, sample_noise(i)))
        p.postprocess_weights(optimizer, weight, grad)
        if i == 1:
          mask_after_update = optimizer.get_slot(weight, 'mask').read_value()
          after_sparsity = tf.math.count_nonzero(mask_after_update, dtype=tf.int32) / tf.size(mask_before_update)
          expected_iter1 = {'before': mask_before_update, 'after': mask_after_update, 
                            'sparsity': after_sparsity}
        elif i == 3:
          mask_after_update = optimizer.get_slot(weight, 'mask').read_value()
          after_sparsity = tf.math.count_nonzero(mask_after_update, dtype=tf.int32) / tf.size(mask_before_update3)
          expected_iter3 = {'before': mask_before_update3, 'after': mask_after_update, 
                            'sparsity': after_sparsity}
        optimizer.iterations.assign_add(1)
      return expected_iter1, expected_iter3

    expected1, expected2 = _train(optimizer, weight)
    self.assertAllEqual(expected1['sparsity'], self.target_sparsity)
    self.assertAllEqual(expected2['sparsity'], self.target_sparsity)
    # assert that sparsity does not change
    self.assertAllEqual(tf.reduce_sum(expected1['before']), tf.reduce_sum(expected1['after']))
    self.assertAllEqual(tf.reduce_sum(expected2['before']), tf.reduce_sum(expected2['after']))
    # assert there are some changes in the mask during each update
    self.assertNotAllEqual(expected1['before'], expected1['after'])
    self.assertNotAllEqual(expected2['before'], expected2['after'])
    del expected1, expected2

    # TODO(xwinxu): test inside a tf.function
    # weight2 = tf.Variable(np.linspace(1.0, 100.0, 100))
    # weight2_dtype = weight2.dtype.base_dtype
    # mask = tf.Variable(
    #     tf.ones(weight2.get_shape(), dtype=weight_dtype),
    #     dtype=weight_dtype)
    # grad = self.grad(weight2)
    # sparse_vars = [(mask, weight2, grad)]
    # _optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    # optimizer = pruning_optimizer.PruningOptimizer(_optimizer, sparsity_config)
    # optimizer.iterations.assign(0)
    # p.create_slots(optimizer, weight2)
    # expected1, expected2 = tf.function(_train)(optimizer, weight2)
    # self.assertAllEqual(expected1['sparsity'], self.target_sparsity)
    # self.assertAllEqual(expected2['sparsity'], self.target_sparsity)
    # # assert that sparsity does not change
    # self.assertAllEqual(tf.reduce_sum(expected1['before']), tf.reduce_sum(expected1['after']))
    # self.assertAllEqual(tf.reduce_sum(expected2['before']), tf.reduce_sum(expected2['after']))
    # # assert there are some changes in the mask during each update
    # self.assertNotAllEqual(expected1['before'], expected1['after'])
    # self.assertNotAllEqual(expected2['before'], expected2['after'])

  @parameterized.parameters(
     ('random_normal',), ('zeros',)
  )
  def testZeroInitGrownConnections(self, reinit_method):
    # new connections are initialized to zeros
    weight = tf.Variable(np.linspace(1.0, 100.0, 100))
    weight_dtype = weight.dtype.base_dtype
    mask = tf.Variable(
        tf.ones(weight.get_shape(), dtype=weight_dtype),
        dtype=weight_dtype)
    grad = self.grad(weight)
    sparse_vars = [(mask, weight, grad)]
    drop_ratio = 0.5
    sparsity_params = {
      'pruning_schedule': self.updater(drop_ratio, 0, 4, 1)
    }
    sparsity_config = pruning_config.LowMagnitudePruningConfig(**sparsity_params) # dummy

    p = pruner.RiGLPruner(
      update_schedule=self.updater(drop_ratio, 0, 4, 4), # update iter 1 and 3
      sparsity=self.target_sparsity,
      block_size=self.block_size,
      block_pooling_type=self.block_pooling_type,
      seed=self.seed,
      noise_std=self.noise_std,
      reinit=reinit_method
    )

    optimizer = pruning_optimizer.PruningOptimizer(
      tf.keras.optimizers.SGD(learning_rate=0.01), sparsity_config)
    optimizer.iterations.assign(0)

    p.create_slots(optimizer, weight)
   
    def weight_mask_op(pruning_vars):
      values_and_vars = []
      for mask, weight, _ in  pruning_vars:
        # weight.assign(tf.math.multiply(weight, mask))
        return tf.math.multiply(weight, mask)

    def train(optimizer, weight, sparse_vars):
      # expected = tf.zeros_like(weight, dtype=weight.dtype.base_dtype) * -1
      expected1 = []
      expected2 = []
      # iterate until all mask updates are complete
      for i in tf.range(4 + 1):
        mask_before_update = optimizer.get_slot(weight, 'mask').read_value()
        weight_before_update = weight.read_value()
        before_sparsity = tf.math.count_nonzero(mask_before_update, dtype=tf.int32) / tf.size(mask_before_update)
        grad = p.preprocess_weights(optimizer, weight, self.grad(weight))
        weight.assign(tf.math.add(weight, sample_noise(i)))
        grad.assign(tf.math.add(grad, sample_noise(i)))
        p.postprocess_weights(optimizer, weight, grad)
        mask_after_update = optimizer.get_slot(weight, 'mask').read_value()
        weight_after_update = weight_mask_op(sparse_vars)
        after_sparsity = tf.math.count_nonzero(mask_after_update, dtype=tf.int32) / tf.size(mask_before_update)
        # sum the values of the new connections
        regrown_indices = tf.math.logical_and(mask_before_update == 0, mask_after_update == 1)
        new_weights1 = weight_after_update[regrown_indices]
        new_weights2 = weight_after_update[regrown_indices]
        if reinit_method == 'zeros':
          expected1.append(tf.math.reduce_all(new_weights1 == 0))
        elif reinit_method == 'random_normal':
          expected2.append(tf.math.reduce_all(new_weights2 != 0))
        optimizer.iterations.assign_add(1)
      
      return [bool(j) for j in expected1] if reinit_method == 'zeros' else [bool(j) for j in expected2]

    expected = train(optimizer, weight, sparse_vars)
    if reinit_method == 'zeros':
      self.assertTrue(np.all(expected) == True)
    elif reinit_method == 'random_normal':
      self.assertTrue(np.all(expected) == False)

    # TODO: tf.function
    weight = tf.Variable(np.linspace(1.0, 100.0, 100))
    weight_dtype = weight.dtype.base_dtype
    mask = tf.Variable(
        tf.ones(weight.get_shape(), dtype=weight_dtype),
        dtype=weight_dtype)
    grad = self.grad(weight)
    sparse_vars = [(mask, weight, grad)]
    sparsity_params = {
      'pruning_schedule': self.updater(drop_ratio, 0, 4, 1)
    }
    sparsity_config = pruning_config.LowMagnitudePruningConfig(**sparsity_params) # dummy
    optimizer = pruning_optimizer.PruningOptimizer(
      tf.keras.optimizers.SGD(learning_rate=0.01), sparsity_config)
    optimizer.iterations.assign(0)

    p.create_slots(optimizer, weight)
    
    expected_tf_function = tf.function(train)(optimizer, weight, sparse_vars)
    if reinit_method == 'zeros':
      self.assertTrue(np.all(expected_tf_function) == True)
    elif reinit_method == 'random_normal':
      self.assertTrue(np.all(expected_tf_function) == False)

  # @parameterized.parameters(
  #   ('ones',), ('zero',), (None,), (0,)
  # )
  # def testValueErrorGetGrowTensor(self, grow_init_method):
  #   # new connections are initialized to zeros
  #   weight = tf.Variable(np.linspace(1.0, 100.0, 100))
  #   weight_dtype = weight.dtype.base_dtype
  #   mask = tf.Variable(
  #       tf.ones(weight.get_shape(), dtype=weight_dtype),
  #       dtype=weight_dtype)
  #   grad = self.grad(weight)
  #   sparse_vars = [(mask, weight, grad)]
  #   drop_ratio = 0.5
  #   sparsity_params = {
  #     'pruning_schedule': self.updater(drop_ratio, 0, 4, 1)
  #   }
  #   sparsity_config = pruning_config.LowMagnitudePruningConfig(**sparsity_params) # dummy

  #   with self.assertRaises(ValueError):
  #     p = pruner.RiGLPruner(
  #       update_schedule=self.updater(drop_ratio, 0, 4, 4), # update iter 1 and 3
  #       sparsity=self.target_sparsity,
  #       block_size=self.block_size,
  #       block_pooling_type=self.block_pooling_type,
  #       seed=self.seed,
  #       noise_std=self.noise_std,
  #       reinit=self.reinit,
  #       grow_init=grow_init_method,
  #     )

if __name__ == "__main__":
  test.main()
