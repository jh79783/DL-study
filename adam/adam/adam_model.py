import class_model.mathutil as mu
import numpy as np


class AdamModel:
    def __init__(self, use_adam=True):
        self.use_adam = use_adam

    def update_param(self, pm, key, delta,learning_rate = 0.001):
        if self.use_adam:  # True 이면 아담 업데이트 시작
            delta = self.eval_adam_delta(pm, key, delta)
        pm[key] -= learning_rate * delta

    def eval_adam_delta(self, pm, key, delta):
        ro_1 = 0.9
        ro_2 = 0.999
        epsilon = 1.0e-8

        skey, tkey, step = 's' + key, 't' + key, 'n' + key
        if skey not in pm:
            pm[skey] = np.zeros(pm[key].shape)
            pm[tkey] = np.zeros(pm[key].shape)
            pm[step] = 0

        s = pm[skey] = ro_1 * pm[skey] + (1 - ro_1) * delta
        t = pm[tkey] = ro_2 * pm[tkey] + (1 - ro_2) * (delta * delta)

        pm[step] += 1
        s = s / (1 - np.power(ro_1, pm[step]))
        t = t / (1 - np.power(ro_2, pm[step]))

        return s / (np.sqrt(t) + epsilon)
