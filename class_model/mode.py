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
    def mode_forward_postproc(self, feature, train_y):
        entropy = mu.softmax_cross_entropy_with_logits(train_y, feature)
        loss = np.mean(entropy)
        aux = [feature, train_y, entropy]

        return loss, aux

    def mode_backprop_postproc(self, G_loss, aux):
        feature, train_y, entropy = aux

        g_loss_entropy = 1.0 / np.prod(entropy.shape)
        g_entropy_output = mu.softmax_cross_entropy_with_logits_derv(train_y, feature)

        G_entropy = g_loss_entropy * G_loss
        G_output = g_entropy_output * G_entropy

        return G_output

    def eval_accuracy(self, train_x, train_y, feature):
        estimate = np.argmax(feature, axis=1)
        answer = np.argmax(train_y, axis=1)
        correct = np.equal(estimate, answer)
        accuracy = np.mean(correct)

        return accuracy

    def get_estimate(self, output):
        estimate = mu.softmax(output)
        return estimate


class Office_Select(ModeBase):
    def __init__(self, cnts):
        self.cnts = cnts

    def mode_forward_postproc(self,feature, train_y):
        features, ys = np.hsplit(feature, self.cnts), np.hsplit(train_y, self.cnts)
        losses = list()
        auxs = list()
        for i in range(2):
            entropy = mu.softmax_cross_entropy_with_logits(ys[i], features[i])

            # print(f"feature:{features[i].shape}")
            loss = np.mean(entropy)
            aux = [features[i], ys[i], entropy]
            losses.append(loss)
            auxs.append(aux)


        return losses, auxs
        # # print("office dataset_forward_postproc")
        #
        #
        # loss0, aux0 = self.mode_forward_postproc(outputs[0], ys[0])
        # loss1, aux1 = self.mode_forward_postproc(outputs[1], ys[1])
        # return loss0 + loss1, [aux0, aux1]

    def mode_backprop_postproc(self, G_loss, aux):
        G_outputs =list()
        for i in range(2):
            feature, train_y, entropy = aux[i]

            g_loss_entropy = 1.0 / np.prod(entropy.shape)
            g_entropy_output = mu.softmax_cross_entropy_with_logits_derv(train_y, feature)

            G_entropy = g_loss_entropy * G_loss
            G_output = g_entropy_output * G_entropy
            G_outputs.append(G_output)

        G_outputs0 = G_outputs[0]
        G_outputs1 = G_outputs[1]
        return np.hstack([G_outputs0, G_outputs1])

    def eval_accuracy(self, train_x, train_Y, feature):

        accs = list()
        outputs, ys = np.hsplit(feature, self.cnts), np.hsplit(train_Y, self.cnts)
        for i in range(2):
            estimate = np.argmax(outputs[i], axis=1)
            answer = np.argmax(ys[i], axis=1)
            correct = np.equal(estimate, answer)
            accuracy = np.mean(correct)
            accs.append(accuracy)

        # acc0 = self.eval_accuracy(x, ys[0], outputs[0])
        # acc1 = self.eval_accuracy(x, ys[1], outputs[1])
        # print(acc0,acc1)
        return accs

    def get_estimate(self, output):
        estimate = mu.softmax(output)
        return estimate