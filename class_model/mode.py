import numpy as np
import class_model.mathutil as mu
from class_model.mode_base import ModeBase


class Regression(ModeBase):
    def mode_forward_postproc(self, output, y):
        diff = output - y
        square = np.square(diff)
        loss = np.mean(square)
        aux = diff

        return loss, aux

    def mode_backprop_postproc(self, G_loss, aux):
        diff = aux
        shape = diff.shape

        g_loss_square = np.ones(shape) / np.prod(shape)
        g_square_diff = 2 * diff
        g_diff_output = 1

        G_square = g_loss_square * G_loss
        G_diff = g_square_diff * G_square
        G_output = g_diff_output * G_diff

        return G_output

    def eval_accuracy(self, x, y, output):
        mse = np.mean(np.square(output - y))
        accuracy = 1 - np.sqrt(mse) / np.mean(y)
        return accuracy

    def get_estimate(self, output):
        estimate = output
        return estimate


class Binary(ModeBase):
    def mode_forward_postproc(self, output, y):
        entropy = mu.sigmoid_cross_entropy_with_logits(y, output)
        loss = np.mean(entropy)
        aux = [y, output]

        return loss, aux

    def mode_backprop_postproc(self, G_loss, aux):
        y, output = aux
        shape = output.shape

        g_loss_entropy = np.ones(shape) / np.prod(shape)
        g_entropy_output = mu.sigmoid_cross_entropy_with_logits_derv(y, output)

        G_entropy = g_loss_entropy * G_loss
        G_output = g_entropy_output * G_entropy

        return G_output

    def eval_accuracy(self, x, y, output):
        estimate = np.greater(output, 0)
        answer = np.equal(y, 1.0)
        correct = np.equal(estimate, answer)
        accuracy = np.mean(correct)

        return accuracy

    def get_estimate(self, output):
        estimate = mu.sigmoid(output)
        return estimate


class Select(ModeBase):
    def mode_forward_postproc(self, output, y):
        entropy = mu.softmax_cross_entropy_with_logits(y, output)
        loss = np.mean(entropy)
        aux = [output, y, entropy]

        return loss, aux

    def mode_backprop_postproc(self, G_loss, aux):
        output, y, entropy = aux

        g_loss_entropy = 1.0 / np.prod(entropy.shape)
        g_entropy_output = mu.softmax_cross_entropy_with_logits_derv(y, output)

        G_entropy = g_loss_entropy * G_loss
        G_output = g_entropy_output * G_entropy

        return G_output

    def eval_accuracy(self, x, y, output):
        estimate = np.argmax(output, axis=1)
        answer = np.argmax(y, axis=1)
        correct = np.equal(estimate, answer)
        accuracy = np.mean(correct)

        return accuracy

    def get_estimate(self, output):
        estimate = mu.softmax(output)
        return estimate