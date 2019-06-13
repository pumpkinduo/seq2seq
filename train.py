import tensorflow as tf
from onedata_util import loadDataset,getBatches
from initialmodel import Seq2SeqModel

from tqdm import tqdm
import math
import os
import time

start = time.clock()

tf.app.flags.DEFINE_integer("rnn_size",200,"Number of hidden units in each layer")
tf.app.flags.DEFINE_integer("batch_size",200,"Batch Size")
tf.app.flags.DEFINE_integer("embedding_size",150,"Embedding dimensions of encoder and decoder inputs")
tf.app.flags.DEFINE_float("learning_rate",0.00001,"Learning rate")
tf.app.flags.DEFINE_integer("num_layers",2,"Number of layers in each encoder and decoder")
tf.app.flags.DEFINE_integer("numEpochs",20,"Maximum # of training epochs")
tf.app.flags.DEFINE_integer("steps_per_checkpoint",5,"Save model checkpoint every this iteration")
tf.app.flags.DEFINE_string("model_dir","titleinitial_state/model/train","Path to save model checkpoints")
tf.app.flags.DEFINE_string("model_name","keyphrase.ckpt","File name used for model checkpoints")

FLAGS = tf.app.flags.FLAGS


int_to_vocab,vocab_to_int,data = loadDataset()

with tf.Session() as sess:
    model = Seq2SeqModel(FLAGS.rnn_size,FLAGS.num_layers,FLAGS.embedding_size,
                         FLAGS.learning_rate,vocab_to_int,mode="train",use_attention=True,
                         beam_search=False,beam_size=20,max_gradient_norm=1.0)

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("reloading model parameters....")
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("create new model parameters...")
        sess.run(tf.global_variables_initializer())
    current_step = 0

    summary_writer = tf.summary.FileWriter(FLAGS.model_dir,graph=sess.graph)
    for e in range(FLAGS.numEpochs):
        print("Epoch{}/{}-------------".format(e+1,FLAGS.numEpochs))
        batches = getBatches(data,FLAGS.batch_size)
      
        for nexBatch in tqdm(batches,desc = "Training"):
            loss,summary = model.train(sess,nexBatch)
            current_step+=1

            if current_step % FLAGS.steps_per_checkpoint == 0:
              
                perplexity = math.exp(float(loss)) if loss<300 else float("inf")
                tqdm.write("----Step %d -- loss %.2f -- Perplexity %.2f" %(current_step,loss,perplexity))
         
                summary_writer.add_summary(summary,current_step)
                checkpoint_path = os.path.join(FLAGS.model_dir,FLAGS.model_name)
                model.saver.save(sess,checkpoint_path,global_step=current_step)
elapsed = (time.clock() - start)
print("Time used:",elapsed)
