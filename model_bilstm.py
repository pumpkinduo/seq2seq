import tensorflow as tf
from tensorflow.python.util import nest

class  Seq2SeqModel():
    def __init__(self,rnn_size,num_layers,embedding_size,learning_rate,word_to_idx,mode,use_attention,
                 beam_search,embedding_matrix,beam_size,max_gradient_norm=5.0):
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.word_to_idx = word_to_idx
        self.vocab_size = len(self.word_to_idx)
        self.mode = mode
        self.use_attention = use_attention
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.max_gradient_norm = max_gradient_norm
        self.embedding_matrix = embedding_matrix
        self.build_model()



    def _create_rnn_cell(self):
        def single_rnn_cell():
            single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size)

            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
            return cell

        cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell
    def build_model(self):
        print("building model.........")
        self.encoder_inputs = tf.placeholder(tf.int32,[None,None],name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32,[None],name='encoder_inputs_length')
        self.batch_size = tf.placeholder(tf.int32,[],name='batch_size')
        self.keep_prob_placeholder = tf.placeholder(tf.float32,name='keep_prob_placeholder')
        self.decoder_targets = tf.placeholder(tf.int32,[None,None],name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32,[None],name='decoder_targets_length')
        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length,name='max_target_len')
        self.mask = tf.sequence_mask(self.decoder_targets_length,self.max_target_sequence_length,dtype=tf.float32,name='masks')

        with tf.variable_scope("encoder"):
            encoder_cell = self._create_rnn_cell()
            encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix,self.encoder_inputs)
            encoder_inputs_embedded = tf.cast(encoder_inputs_embedded,tf.float32)

            outputs,encoder_state= tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,cell_bw=encoder_cell,inputs=encoder_inputs_embedded,sequence_length=self.encoder_inputs_length,dtype=tf.float32)
       
            encoder_outputs = tf.concat(outputs, axis=2)
            encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2])

        with tf.variable_scope("decoder"):
            encoder_inputs_length = self.encoder_inputs_length
            #beam search
            if self.beam_search:
                print("use beam search decoding...")

                encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs,multiplier=self.beam_size)
                encoder_state = nest.map_structure(lambda s:tf.contrib.seq2seq.tile_batch(s,self.beam_size),encoder_state)
                encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.encoder_inputs_length,multiplier=self.beam_size)

        
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size,memory=encoder_outputs,memory_sequence_length=encoder_inputs_length)

            # attention wrapper
            decoder_cell = self._create_rnn_cell()
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,attention_mechanism=attention_mechanism,
                                                               attention_layer_size=self.rnn_size,name="Attention_Wrapper")

            batch_size = self.batch_size if not self.beam_search else self.batch_size*self.beam_size

            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size,dtype= tf.float32).clone(cell_state=encoder_state[0])
            output_layer = tf.layers.Dense(self.vocab_size,kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))

            if self.mode == "train":
                ending = tf.strided_slice(self.decoder_targets,[0,0],[self.batch_size,-1],[1,1])
                decoder_input = tf.concat([tf.fill([self.batch_size,1],self.word_to_idx["<GO>"]),ending],1)
                decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix,decoder_input)
                decoder_inputs_embedded = tf.cast(decoder_inputs_embedded,tf.float32)
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                                    sequence_length=self.decoder_targets_length,
                                                                    time_major=False,name="training_helper")
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell,helper= training_helper,
                                                                   initial_state= decoder_initial_state,
                                                                   output_layer=output_layer)
              
                decoder_outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder= training_decoder,
                                                                        impute_finished= True,
                                                                        maximum_iterations= self.max_target_sequence_length)

    p
                self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
                self.decoder_predict_train = tf.argmax(self.decoder_logits_train,axis=-1,name="decoder_pred_train")

                self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                             targets=self.decoder_targets,
                                                             weights=self.mask)

  
                tf.summary.scalar("loss",self.loss)
       
                self.summary_op = tf.summary.merge_all()

                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss,trainable_params)
                clip_gradients,_ = tf.clip_by_global_norm(gradients,self.max_gradient_norm)
                self.train_op = optimizer.apply_gradients(zip(clip_gradients,trainable_params))

            elif self.mode == "decode":
 
                start_tokens = tf.ones([self.batch_size,],tf.int32) * self.word_to_idx["<GO>"]

                end_token = self.word_to_idx["<EOS>"]
                if self.beam_search:
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell= decoder_cell,embedding= self.embedding_matrix,
                                                                             start_tokens=start_tokens,
                                                                             end_token=end_token,
                                                                             initial_state=decoder_initial_state,
                                                                             beam_width= self.beam_size,
                                                                             output_layer=output_layer)
                else:
                    decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embedding_matrix,
                                                                               start_tokens=start_tokens,
                                                                               end_token=end_token)
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,helper=decoding_helper,
                                                                        initial_state=decoder_initial_state,
                                                                        output_layer= output_layer)
                decoder_outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder= inference_decoder,
                                                                        maximum_iterations=10)
          
                if self.beam_search:
                    self.decoder_predict_decode = decoder_outputs.predicted_ids
                else:
           
                    self.decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id,-1)
        print("save model...............")
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self,sess,batch):
        feed_dict = {self.encoder_inputs:batch.encoder_inputs,
                     self.encoder_inputs_length:batch.encoder_inputs_length,
                     self.decoder_targets:batch.decoder_targets,
                     self.decoder_targets_length:batch.decoder_targets_length,
                     self.keep_prob_placeholder:1.0,
                     self.batch_size:len(batch.encoder_inputs)}
        _,loss,summary = sess.run([self.train_op,self.loss,self.summary_op],feed_dict=feed_dict)
        return loss,summary

    def eval(self,sess,batch):

        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.decoder_targets: batch.decoder_targets,
                     self.decoder_targets_length: batch.decoder_targets_length,
                     self.keep_prob_placeholder: 1.0,
                     self.batch_size: len(batch.encoder_inputs)}
        loss,summary = sess.run([self.loss,self.summary_op],feed_dict=feed_dict)
        return loss,summary
    def infer(self,sess,batch):

        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.keep_prob_placeholder: 1.0,
                     self.batch_size: len(batch.encoder_inputs)}
        predict = sess.run([self.decoder_predict_decode],feed_dict=feed_dict)
        return predict

