"""Run inside the pinned TensorFlow/StellarGraph image, separately from audit tests."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import stellargraph as sg
from stellargraph.mapper import HinSAGENodeGenerator
from stellargraph.layer import HinSAGE
import tensorflow as tf
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "competitors/scripts"))
from graph_protocol import isolated_queries, training_graph
from keras_metrics import GlobalR2
from keras_callbacks import RestoreBestAtEnd


def graph(second_feature=2.):
    return sg.StellarGraph({
        'interaction': pd.DataFrame([[1.], [second_feature]], index=['i1', 'i2']),
        'siRNA': pd.DataFrame([[1.], [2.]], index=['s1', 's2']),
        'mRNA': pd.DataFrame([[1.]], index=['m'])},
        edges=pd.DataFrame({'source':['i1','i1','i2','i2'], 'target':['s1','m','s2','m']}))


class GraphProtocolChecks(unittest.TestCase):
    def test_training_graph_has_no_validation_interaction(self):
        fit = training_graph(graph(), ['i1'])
        self.assertEqual(set(fit.nodes()), {'i1','s1','m'})

    def test_evaluation_sequence_nodes_are_private_to_each_query(self):
        isolated = isolated_queries(graph(), ['i1','i2'])
        self.assertEqual(isolated.number_of_nodes(), 6)
        for kind in ['siRNA', 'mRNA']:
            for node in isolated.nodes(node_type=kind):
                self.assertEqual(len(isolated.neighbors(node)), 1)

    def test_hinsage_prediction_is_invariant_to_other_evaluation_samples(self):
        tf.random.set_seed(42)
        reference = isolated_queries(graph(), ['i1','i2'])
        generator = HinSAGENodeGenerator(reference, 2, [8,4], head_node_type='interaction')
        layer = HinSAGE(layer_sizes=[4,2], generator=generator, bias=True, dropout=0)
        inputs, embeddings = layer.in_out_tensors()
        model = tf.keras.Model(inputs=inputs, outputs=tf.keras.layers.Dense(1)(embeddings))
        baseline = model.predict(generator.flow(['i1','i2']), verbose=0)[0]
        changed = isolated_queries(graph(1e6), ['i1','i2'])
        altered = HinSAGENodeGenerator(changed, 2, [8,4], head_node_type='interaction')
        np.testing.assert_allclose(baseline, model.predict(altered.flow(['i1','i2']), verbose=0)[0], atol=1e-6)
        alone = isolated_queries(graph(), ['i1'])
        single = HinSAGENodeGenerator(alone, 1, [8,4], head_node_type='interaction')
        np.testing.assert_allclose(baseline, model.predict(single.flow(['i1']), verbose=0)[0], atol=1e-6)

    def test_r2_is_global_across_unequal_and_constant_batches(self):
        truth = np.array([.1,.1,.7,.9,1.])
        predictions = np.array([.2,.4,.6,.8,.9])
        metric = GlobalR2()
        metric.update_state(truth[:2], predictions[:2])
        metric.update_state(truth[2:], predictions[2:])
        self.assertAlmostEqual(float(metric.result()), r2_score(truth, predictions), places=10)
        metric.reset_states()
        metric.update_state(truth, predictions)
        self.assertAlmostEqual(float(metric.result()), r2_score(truth, predictions), places=10)

    def test_best_checkpoint_is_restored_at_epoch_cap(self):
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(1,)), tf.keras.layers.Dense(1, use_bias=False)])
        callback = RestoreBestAtEnd(monitor="val_r2_metric", mode="max", patience=20, restore_best_weights=True)
        callback.set_model(model)
        callback.on_train_begin()
        for epoch, (weight, metric) in enumerate([(1., .8), (2., .7)]):
            model.set_weights([np.array([[weight]], dtype=np.float32)])
            callback.on_epoch_end(epoch, {"val_r2_metric": metric})
        callback.on_train_end()
        self.assertEqual(float(model.get_weights()[0][0,0]), 1.)

    def test_keras_evaluation_reports_global_r2(self):
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(1,)), tf.keras.layers.Dense(1, use_bias=False, kernel_initializer='ones')])
        model.compile(optimizer='sgd', loss='mse', metrics=[GlobalR2()])
        truth = np.array([.1,.1,.7,.9,1.], dtype=np.float32)
        predictions = np.array([.2,.4,.6,.8,.9], dtype=np.float32)
        values = model.evaluate(predictions, truth, batch_size=2, verbose=0, return_dict=True)
        self.assertAlmostEqual(values['r2_metric'], r2_score(truth, predictions), places=6)
        restored = tf.keras.metrics.deserialize(tf.keras.metrics.serialize(GlobalR2()))
        self.assertIsInstance(restored, GlobalR2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
