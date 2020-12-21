import tensorflow as tf
from tacotron.utils.symbols import symbols
from tacotron.models.helpers import TacoTestHelper
from tacotron.models.modules import *
from tensorflow.contrib.seq2seq import dynamic_decode
from tacotron.models.Architecture_wrappers import TacotronEncoderCell, TacotronDecoderCell
from tacotron.models.custom_decoder import CustomDecoder
from tacotron.models.attention import LocationSensitiveAttention


class Tacotron():
	def __init__(self, hparams):
		self._hparams = hparams

	def initialize(self, inputs, input_lengths, mel_targets=None, stop_token_targets=None):

		with tf.variable_scope('inference') as scope:
			is_training = False  # No entrenando
			batch_size = tf.shape(inputs)[0]
			hp = self._hparams

			# Tabla de embeddings de speaker
			embedding_table = tf.get_variable('inputs_embedding', [len(symbols), hp.embedding_dim], dtype=tf.float32)

			#
			embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)

			# Encoder: Capas convolucionales y LSTM
			encoder_cell = TacotronEncoderCell(
				EncoderConvolutions(
					is_training,
					kernel_size=(5, ),
					channels=512,
					scope='encoder_convolutions'
				),
				EncoderRNN(
					is_training,
					size=hp.encoder_lstm_units,
					zoneout=hp.tacotron_zoneout_rate,
					scope='encoder_LSTM')
			)

			# Salida Encoder
			encoder_outputs = encoder_cell(embedded_inputs, input_lengths)

			# Decoder ### Definir elementos para el Decoder

			# Prenet: Dos capas de 256 unidades ReLU
			prenet = Prenet(is_training, layer_sizes=[256, 256], scope='decoder_prenet')

			# Red Atencion
			print("PARAMS ATTENTION", hp.attention_dim, encoder_outputs)
			attention_mechanism = LocationSensitiveAttention(
				hp.attention_dim, encoder_outputs
			)

			# Decoder LSTM
			decoder_lstm = DecoderRNN(
				is_training,
				layers=hp.decoder_layers,
				size=hp.decoder_lstm_units,
				zoneout=hp.tacotron_zoneout_rate,
				scope='decoder_lstm'
			)

			# Frame projection: proyectar resultados a 80 mels
			frame_projection = FrameProjection(hp.num_mels * hp.outputs_per_step, scope='linear_transform')

			# Stop projection: Projectar salida usando token para separar las palabras individuales
			stop_projection = StopProjection(is_training, scope='stop_token_projection')

			# Unir todo
			decoder_cell = TacotronDecoderCell(
				prenet,
				attention_mechanism,
				decoder_lstm,
				frame_projection,
				stop_projection,
				mask_finished=hp.mask_finished)

			# self.helper = TacoTrainingHelper(batch_size, mel_targets, stop_token_targets, hp.num_mels, hp.outputs_per_step, hp.tacotron_teacher_forcing_ratio)
			# Modo de sintesis
			self.helper = TacoTestHelper(batch_size, hp.num_mels, hp.outputs_per_step)
			# Poner Decoder en estado inicial - zero state
			decoder_init_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
			max_iters = hp.max_iters if not is_training else None
			(frames_prediction, stop_token_prediction, _), final_decoder_state, _ = dynamic_decode(
				CustomDecoder(decoder_cell, self.helper, decoder_init_state),
				impute_finished=hp.impute_finished,
				maximum_iterations=max_iters)

			# Reshape outputs: 1 output por entrada
			decoder_output = tf.reshape(frames_prediction, [batch_size, -1, hp.num_mels])
			stop_token_prediction = tf.reshape(stop_token_prediction, [batch_size, -1])

			# Postnet: cinco capas convolucionales
			postnet = Postnet(
				is_training,
				kernel_size=hp.postnet_kernel_size,
				channels=hp.postnet_channels,
				scope='postnet_convolutions'
			)

			# Resultados
			results = postnet(decoder_output)

			# Proyectar resultados a 80 mels
			results_projection = FrameProjection(hp.num_mels, scope='postnet_projection')
			projected_results = results_projection(results)

			# Calcular espectrograma mel
			mel_outputs = decoder_output + projected_results

			# Tomar alineacion del ultimo estado del decoder
			alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

			self.inputs = inputs
			self.input_lengths = input_lengths
			self.decoder_output = decoder_output
			self.alignments = alignments
			self.stop_token_prediction = stop_token_prediction
			self.stop_token_targets = stop_token_targets
			self.mel_outputs = mel_outputs
			self.mel_targets = mel_targets

	def add_loss(self):
		with tf.variable_scope('loss') as scope:
			# Antes postnet
			before = tf.losses.mean_squared_error(self.mel_targets, self.decoder_output)
			# Despues post net
			after = tf.losses.mean_squared_error(self.mel_targets, self.mel_outputs)

			self.loss = before + after

	def add_optimizer(self, global_step):
		# TODO
		print('hola')

	def _learning_rate_decay(self, init_lr, global_step):
		# TODO
		return lr
