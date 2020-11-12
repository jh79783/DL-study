class ModeBase:
    def mode_forward_postproc(self, output, y):
        pass

    def mode_backprop_postproc(self, G_loss, aux):
        pass

    def eval_accuracy(self, x, y, output):
        pass

    def get_estimate(self, output):
        pass