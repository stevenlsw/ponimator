import math
import time
import numpy as np


def print_current_loss(start_time, niter_state, losses, epoch=None, inner_iter=None, lr=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None and lr is not None :
        print('epoch: %3d niter:%6d inner_iter:%4d lr:%5f' % (epoch, niter_state, inner_iter, lr), end=" ")
    elif epoch is not None:
        print('epoch: %3d niter:%6d inner_iter:%4d' % (epoch, niter_state, inner_iter), end=" ")

    now = time.time()
    message = '%s'%(as_minutes(now - start_time))

    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    print(message)


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new
