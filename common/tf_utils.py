import tensorflow as tf

def soft_copy(session, target, source, tau):
    op = []
    for t, s in zip(target, source):
        op.append(tf.assign(t, tau * s + (1. - tau) * t)) # soft target update
    return op
