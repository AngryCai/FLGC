from FLGC import SemiFLGC
from utils import load_data, cal_test_acc
import numpy as np
import time

# # Cora, PubMed, CiteSeer
data_name = 'Cora'  # # settings: SGC: regularization_coef=2, K_hop=2; APPNP: alpha=0.1, K=20
# data_name = 'CiteSeer'  # # settings: SGC: regularization_coef=30, K_hop=2; APPNP: alpha=0.1, K=20
# data_name = 'PubMed'  # # settings: regularization_coef=0.02, K_hop=2; APPNP: alpha=0.05, K=15

root = './tmp/' + data_name
score_acc = []
time_list = []
for i in range(1):
    data = load_data(root, data_name, split='public')
    # data.train_mask = ~(data.test_mask + data.val_mask)

    print('num-train: %s, num-val: %s, num-test: %s' %
          (data.train_mask.sum().item(), data.val_mask.sum().item(), data.test_mask.sum().item()))
    start_time = time.time()
    model = SemiFLGC(agg='appnp', regularization_coef=2, K_hop=20, alpha=0.1).to('cuda:0')
    y_pred = model(data)
    run_time = time.time() - start_time
    acc_val = cal_test_acc(data.y_one_hot[data.val_mask], y_pred[data.val_mask]).item()
    acc_test = cal_test_acc(data.y_one_hot[data.test_mask], y_pred[data.test_mask]).item()

    print('ACC: %.4f' % acc_test)
    score_acc.append(acc_test)
    time_list.append(run_time)

print('===========================================')
print('ACC: {:.2f} +- {:.2f}'.format(np.mean(score_acc)*100, np.std(score_acc)*100))
print('TIME: {:.4f}s'.format(np.mean(time_list)))
