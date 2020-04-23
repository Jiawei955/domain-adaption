import numpy as np
import os
import re

root = '../raw/'
adaption = ['amazon','dslr']
npyfiles = os.listdir(root)
npyfile = [os.path.join(root,i) for i in npyfiles]
adaptation = "AMAZON_to_DSLR"

# amazon -> dslr
for repetition in range(5):
    for spc in range(1,5):
        X_train_source_list = []
        y_train_source_list = []
        X_train_target_list = []
        y_train_target_list = []
        X_test_target_list = []
        y_test_target_list = []
        for eachfile in npyfile:
            sp = re.split('[^a-zA-Z0-9]', eachfile)

            # print(sp)
            if sp[4] == 'amazon':
                f = np.load(eachfile)
                random_index = np.random.choice(f.shape[0],20,replace=False)
                X_train_source_list.append(f[random_index])
                source_label = int(sp[-2])
                y_train_source_list.append(np.full(20,source_label))
            elif sp[4] == 'dslr':
                f = np.load(eachfile)
                random_index = np.random.choice(f.shape[0], 8, replace=True)
                X_train_target_list.append(f[random_index[0:spc]])
                target_label = int(sp[-2])
                y_train_target_list.append(np.full(spc,target_label))
                X_test_target_list.append(f[random_index[spc:]])
                y_test_target_list.append(np.full(8-spc,target_label))

        X_train_source = X_train_source_list[0]
        for j in range(1, len(X_train_source_list)):
            X_train_source = np.concatenate((X_train_source, X_train_source_list[j]))

        y_train_source = y_train_source_list[0]
        for j in range(1, len(y_train_source_list)):
            y_train_source = np.concatenate((y_train_source, y_train_source_list[j]))

        X_train_target = X_train_target_list[0]
        for j in range(1, len(X_train_target_list)):
            X_train_target = np.concatenate((X_train_target, X_train_target_list[j]))

        y_train_target = y_train_target_list[0]
        for j in range(1,len(y_train_target_list)):
            y_train_target = np.concatenate((y_train_target,y_train_target_list[j]))

        X_test_target = X_test_target_list[0]
        for j in range(1, len(X_test_target_list)):
            X_test_target = np.concatenate((X_test_target, X_test_target_list[j]))

        y_test_target = y_test_target_list[0]
        for j in range(1, len(y_test_target_list)):
            y_test_target = np.concatenate((y_test_target, y_test_target_list[j]))

        print("X_train_target: ", X_train_target.shape)
        print("y_train_target: ", y_train_target.shape)

        print("X_test_target: ", X_test_target.shape)
        print("y_test_target: ", y_test_target.shape)

        # save X_train_source

        filename = f"../amazon_dslr/{adaptation}_X_train_source_repetition_{repetition}_spc_{spc}"
        np.save(filename, X_train_source)

        # save y_train_source

        filename = f"../amazon_dslr/{adaptation}_y_train_source_repetition_{repetition}_spc_{spc}"
        np.save(filename, y_train_source)

        # save X_train_target
        filename = f"../amazon_dslr/{adaptation}_X_train_target_repetition_{repetition}_spc_{spc}"
        np.save(filename, X_train_target)
        # save y_train_target
        filename = f"../amazon_dslr/{adaptation}_y_train_target_repetition_{repetition}_spc_{spc}"
        np.save(filename, y_train_target)

        # save X_test_target
        filename = f"../amazon_dslr/{adaptation}_X_test_target_repetition_{repetition}_spc_{spc}"
        np.save(filename, X_test_target)

        # save y_test_target
        filename = f"../amazon_dslr/{adaptation}_y_test_target_repetition_{repetition}_spc_{spc}"
        np.save(filename, y_test_target)























