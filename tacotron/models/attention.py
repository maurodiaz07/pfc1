import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops


def _compute_attention(attention_mechanism, cell_output, attention_state, attention_layer):

	alignments, next_attention_state = attention_mechanism(cell_output, state=attention_state)

	expanded_alignments = array_ops.expand_dims(alignments, 1)
	context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
	context = array_ops.squeeze(context, [1])
	attention = context

	return attention, alignments, next_attention_state


def _location_sensitive_score(W_query, W_fil, W_keys):
	dtype = W_query.dtype
	num_units = W_keys.shape[-1].value or array_ops.shape(W_keys)[-1]

	v_a = tf.get_variable(
		'attention_variable', shape=[num_units], dtype=dtype,
		initializer=tf.contrib.layers.xavier_initializer())
	b_a = tf.get_variable(
		'attention_bias', shape=[num_units], dtype=dtype,
		initializer=tf.zeros_initializer())

	return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fil + b_a), [2])


class LocationSensitiveAttention(BahdanauAttention):

	def __init__(self, num_units, memory, name='LocationSensitiveAttention'):
		super(LocationSensitiveAttention, self).__init__(
			num_units=num_units,
			memory=memory,
			memory_sequence_length=None,
			probability_fn=None,
			name=name)

		self.location_convolution = tf.layers.Conv1D(
			filters=32,
			kernel_size=(31, ),
			padding='same',
			use_bias=False,
			name='location_features_convolution'
		)
		self.location_layer = tf.layers.Dense(
			units=num_units,
			use_bias=False,
			dtype=tf.float32,
			name='location_features_layer'
		)

	def __call__(self, query, state):
		previous_alignments = state
		with variable_scope.variable_scope(None, "Location_Sensitive_Attention", [query]):

			# processed_query shape [batch_size, query_depth] -> [batch_size, attention_dim]
			processed_query = self.query_layer(query) if self.query_layer else query
			# -> [batch_size, 1, attention_dim]
			processed_query = tf.expand_dims(processed_query, 1)

			# processed_location_features shape [batch_size, max_time, attention dimension]
			# [batch_size, max_time] -> [batch_size, max_time, 1]
			expanded_alignments = tf.expand_dims(previous_alignments, axis=2)
			# location features [batch_size, max_time, filters]
			f = self.location_convolution(expanded_alignments)
			# Projected location features [batch_size, max_time, attention_dim]
			processed_location_features = self.location_layer(f)

			# energy shape [batch_size, max_time]
			energy = _location_sensitive_score(processed_query, processed_location_features, self.keys)

		# alignments shape = energy shape = [batch_size, max_time]
		alignments = self._probability_fn(energy, previous_alignments)
		next_state = alignments + previous_alignments

		return alignments, next_state
