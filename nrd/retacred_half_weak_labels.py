import json
from scipy import spatial
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def main():
    id2rel = {}

    rel2id = json.load(open('retacred_mix_half/rel2id.json', 'r'))
    for key in rel2id:
        id2rel[rel2id[key]] = key
    print(id2rel)

    label_dict = {0: ['o', 'b', 'org:member_of'], 1: ['o', 'g', 'org:top_members/employees'], 2: ['o', 'r', 'per:age'],
                  3: ['o', 'c', 'per:country_of_death'], 4: ['o', 'm', 'per:stateorprovince_of_birth'],
                  5: ['o', 'y', 'per:cause_of_death'], 6: ['o', 'k', 'org:political/religious_affiliation'],
                  7: ['o', 'darkorange', 'per:parents'], 8: ['o', 'pink', 'per:employee_of'], 9: ['v', 'b', 'per:religion'],
                  10: ['v', 'g', 'per:city_of_birth'], 11: ['v', 'r', 'per:charges'], 12: ['v', 'c', 'org:country_of_branch'],
                  13: ['v', 'm', 'org:website'], 14: ['v', 'y', 'per:stateorprovince_of_death'], 15: ['v', 'k', 'org:members'],
                  16: ['v', 'darkorange', 'per:cities_of_residence'], 17: ['v', 'pink', 'org:number_of_employees/members'],
                  18: ['s', 'b', 'org:founded'], 19: ['s', 'g', 'per:children'], 20: ['s', 'r', 'org:shareholders'],
                  21: ['s', 'c', 'per:other_family'], 22: ['s', 'm', 'per:origin'], 23: ['s', 'y', 'per:identity'],
                  24: ['s', 'k', 'org:dissolved'], 25: ['s', 'darkorange', 'org:stateorprovince_of_branch'],
                  26: ['s', 'pink', 'org:alternate_names'], 27: ['+', 'b', 'per:stateorprovinces_of_residence'],
                  28: ['+', 'g', 'per:spouse'], 29: ['+', 'r', 'per:schools_attended'], 30: ['+', 'c', 'per:country_of_birth'],
                  31: ['+', 'm', 'per:title'], 32: ['+', 'y', 'per:date_of_death'], 33: ['+', 'k', 'per:siblings'],
                  34: ['+', 'darkorange', 'org:founded_by'], 35: ['+', 'pink', 'org:city_of_branch'],
                  36: ['x', 'b', 'per:countries_of_residence'], 37: ['x', 'g', 'per:date_of_birth'],
                  38: ['x', 'r', 'per:city_of_death']}
    for key in label_dict:
        label_dict[key][2] = id2rel[key]
    #print(id2rel)
    #print(label_dict)


    sentence_test = json.load(open('retacred_mix_half/retacred_test_sentence_half.json', 'r'))
    sentence_test_label = json.load(open('retacred_mix_half/retacred_test_label_half.json', 'r'))

    s = np.array(sentence_test)
    print(len(s))
    label = np.array(sentence_test_label)

    # relations have been randomly shuffled, the last six are set to be novel relations
    novel_label_lst = [33, 34, 35, 36, 37, 38]
    top_rel_num = 33
    cluster_label_lst = [33, 34, 35, 36, 37, 38]

    novel_sentence_lst = []
    novel_label = []

    lambda_lst = [100]
    for ld in lambda_lst:
        test_S_vectors = json.load(open('retacred_mix_half/retacred_test_S_vectors_half_ld{}.json'.format(ld), 'r'))

        d = np.array(test_S_vectors)
        print(len(d))

        num_total = len(d)
        print(num_total)

        # we want the outlier to be the last 5%
        ratio = 0.05

        # decide threshold
        fst_sim_lst = []
        for i in range(len(d)):
            ori_dict, cur_ori_gt_sim = findTopkwsim(d[i], top_rel_num, 5, label[i])
            ori_fst_sim = 0
            ori_key = 0
            for key in ori_dict:
                ori_fst_sim = ori_dict[key]
                ori_key = key
                break
            fst_sim_lst.append(ori_fst_sim)

        sorted_fst_sim_lst = sorted(fst_sim_lst)
        threshold = sorted_fst_sim_lst[int(len(d)*ratio)+1]
        print("The computed 5% outlier threshold is : {}".format(threshold))


        # keep those instances closest to the corresponding centroid as weak labels
        GMM_prob_filter = 0.95

        o_remain_lst = []
        o_remain_label_lst = []
        o_remain_sentence_lst = []

        novel_num_add = 0
        i_correct_pred = 0
        i_total_pred = 0
        i_num_remain = 0
        i_num_gt_top1 = 0
        i_num_gt_top3 = 0
        i_num_gt_top5 = 0
        e_i_num_remain = 0
        e_i_num_gt_top1 = 0
        e_i_num_gt_top3 = 0
        e_i_num_gt_top5 = 0

        for i in range(len(d)):
            ori_dict, cur_ori_gt_sim = findTopkwsim(d[i], top_rel_num, 5, label[i])

            ori_fst_sim = 0
            ori_key = 0
            rw_fst_sim = 0
            for key in ori_dict:
                ori_fst_sim = ori_dict[key]
                ori_key = key
                break

            # not belongs to remain_intersection
            if ori_fst_sim >= threshold or rw_fst_sim >= threshold:
                i_total_pred += 1
                if label[i] in novel_label_lst:
                    novel_num_add += 1
                if ori_key == str(label[i]):
                    i_correct_pred += 1

            else:
                o_remain_lst.append(d[i])
                o_remain_label_lst.append(label[i])
                o_remain_sentence_lst.append(s[i])

                i_num_remain += 1
                cnt1, cnt2, cnt3 = countRecall(ori_dict, label[i])
                i_num_gt_top1 += cnt1
                i_num_gt_top3 += cnt2
                i_num_gt_top5 += cnt3
                # belongs to existing relations
                if label[i] not in novel_label_lst:
                    e_i_num_remain += 1
                    e_i_num_gt_top1 += cnt1
                    e_i_num_gt_top3 += cnt2
                    e_i_num_gt_top5 += cnt3

        remain_lst = []
        remain_label_lst = []
        remain_sentence_lst = []
        pred_label_lst = []
        num_clusters = len(novel_label_lst)
        pred_labels, mean_vectors, pred_probabilities = GMM(o_remain_lst, num_clusters)
        # if using GMM prob to filter
        if GMM_prob_filter > 0:
            for j in range(len(o_remain_lst)):
                if pred_probabilities[j][pred_labels[j]] >= GMM_prob_filter:
                    remain_lst.append(o_remain_lst[j])
                    remain_label_lst.append(o_remain_label_lst[j])
                    remain_sentence_lst.append(o_remain_sentence_lst[j])
                    pred_label_lst.append(pred_labels[j])
        else:
            for j in range(len(o_remain_lst)):
                remain_lst.append(o_remain_lst[j])
                remain_label_lst.append(o_remain_label_lst[j])
                remain_sentence_lst.append(o_remain_sentence_lst[j])
                pred_label_lst.append(pred_labels[j])

        remain_arr = np.array(remain_lst)
        remain_label_arr = np.array(remain_label_lst)
        remain_sentence_arr = np.array(remain_sentence_lst)
        novel_maj_labels = []
        num_purity_lst = []
        for i in range(num_clusters):
            tmp_label_lst = []
            tmp_sentence_lst = []
            for j in range(len(remain_arr)):
                if pred_label_lst[j] == i:
                    tmp_label_lst.append(remain_label_arr[j])
                    tmp_sentence_lst.append([remain_sentence_arr[j], str(remain_label_arr[j])])
            if len(tmp_label_lst) == 0:
                continue
            majority_label, cnt = computePurity(tmp_label_lst)
            purity = '{:.3f}'.format(cnt/len(tmp_label_lst))
            belongs2novel = False
            if majority_label in novel_label_lst:
                belongs2novel = True
                novel_maj_labels.append(majority_label)

            print("cluster_{}: ".format(i))
            print("majority_label is: {}".format(majority_label) + " ({})".format(id2rel[majority_label]) + ", belongs to novel relation: {}".format(belongs2novel))
            print("cluster purity is: {}".format(purity))
            print("# instances is: {}".format(len(tmp_label_lst)))

            for sen_id in range(len(tmp_sentence_lst)):
                novel_sentence_lst.append(tmp_sentence_lst[sen_id][0])
                novel_label.append(cluster_label_lst[i])

            num_purity_lst.append([len(tmp_label_lst), float(purity)])
            tmp_label_dict = {}
            for l in tmp_label_lst:
                if label_dict[l][2] not in tmp_label_dict:
                    tmp_label_dict[label_dict[l][2]] = 1
                else:
                    tmp_label_dict[label_dict[l][2]] += 1
            print(dict(sorted(tmp_label_dict.items(), key=lambda item: item[1], reverse=True)))
            print()

        print("# found novel relations:")
        print(len(set(novel_maj_labels)))
        print()
        plot_tsne(remain_label_arr.reshape(-1, 1), remain_arr, 'tsne_test_remain_ld{}.png'.format(ld))
        print("# clustered instances:")
        print(len(remain_arr))
        remain_label_dict = {}
        for l in remain_label_lst:
            if label_dict[l][2] not in remain_label_dict:
                remain_label_dict[label_dict[l][2]] = 1
            else:
                remain_label_dict[label_dict[l][2]] += 1
        print(dict(sorted(remain_label_dict.items(), key=lambda item: item[1], reverse=True)))
        print()
        wcp = 0
        for p in num_purity_lst:
            wcp += p[0] * p[1] / (len(remain_arr))
        wcp = '{:.3f}'.format(wcp)
        print("weighted cluster purity is: {}".format(wcp))
        print()

    train_sentences_ori = json.load(open('retacred_mix_half/retacred_train_sentence_half.json', 'r'))
    train_labels_ori = json.load(open('retacred_mix_half/retacred_train_label_half.json', 'r'))
    train_attribute_vector = json.load(open('retacred_mix_half/retacred_train_S_vectors_half_ld100.json', 'r'))
    print(len(train_sentences_ori))
    print(len(train_attribute_vector))
    train_sentences = []
    train_labels = []
    existing_dict = {}
    for i in range(len(train_sentences_ori)):
        if train_labels_ori[i] not in existing_dict:
            existing_dict[train_labels_ori[i]] = [[train_sentences_ori[i], train_attribute_vector[i]]]
        else:
            existing_dict[train_labels_ori[i]].append([train_sentences_ori[i], train_attribute_vector[i]])
    for key in existing_dict:
        key_attribute_vector = [0 for j in range(top_rel_num)]
        key_attribute_vector[key] = 1
        all_sim_dict = {}
        for k in range(len(existing_dict[key])):
            item = existing_dict[key][k]
            cur_sim = 1 - spatial.distance.cosine(key_attribute_vector, item[1])
            all_sim_dict[k] = cur_sim
        sorted_dict = dict(sorted(all_sim_dict.items(), key=lambda item: item[1], reverse=True))

        for key2 in sorted_dict:
            train_sentences.append(existing_dict[key][key2][0])
            train_labels.append(key)


    train_sentences.extend(novel_sentence_lst)
    train_labels.extend(novel_label)


    tmp_outF = open("retacred_mix_half/retacred_train_sentence_half_novel_added.json", "w")
    json.dump(train_sentences, tmp_outF)
    tmp_outF.close()
    tmp_outF2 = open("retacred_mix_half/retacred_train_label_half_novel_added.json", "w")
    json.dump(train_labels, tmp_outF2)
    tmp_outF2.close()



def plot_tsne(test_labels, test_vectors, file_name):
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    test_tsne = tsne.fit_transform(test_vectors)

    label_dict = {0: ['o', 'b', 'org:member_of'], 1: ['o', 'g', 'org:top_members/employees'], 2: ['o', 'r', 'per:age'],
                  3: ['o', 'c', 'per:country_of_death'], 4: ['o', 'm', 'per:stateorprovince_of_birth'],
                  5: ['o', 'y', 'per:cause_of_death'], 6: ['o', 'k', 'org:political/religious_affiliation'],
                  7: ['o', 'darkorange', 'per:parents'], 8: ['o', 'pink', 'per:employee_of'], 9: ['v', 'b', 'per:religion'],
                  10: ['v', 'g', 'per:city_of_birth'], 11: ['v', 'r', 'per:charges'], 12: ['v', 'c', 'org:country_of_branch'],
                  13: ['v', 'm', 'org:website'], 14: ['v', 'y', 'per:stateorprovince_of_death'], 15: ['v', 'k', 'org:members'],
                  16: ['v', 'darkorange', 'per:cities_of_residence'], 17: ['v', 'pink', 'org:number_of_employees/members'],
                  18: ['s', 'b', 'org:founded'], 19: ['s', 'g', 'per:children'], 20: ['s', 'r', 'org:shareholders'],
                  21: ['s', 'c', 'per:other_family'], 22: ['s', 'm', 'per:origin'], 23: ['s', 'y', 'per:identity'],
                  24: ['s', 'k', 'org:dissolved'], 25: ['s', 'darkorange', 'org:stateorprovince_of_branch'],
                  26: ['s', 'pink', 'org:alternate_names'], 27: ['+', 'b', 'per:stateorprovinces_of_residence'],
                  28: ['+', 'g', 'per:spouse'], 29: ['+', 'r', 'per:schools_attended'], 30: ['+', 'c', 'per:country_of_birth'],
                  31: ['+', 'm', 'per:title'], 32: ['+', 'y', 'per:date_of_death'], 33: ['+', 'k', 'per:siblings'],
                  34: ['+', 'darkorange', 'org:founded_by'], 35: ['+', 'pink', 'org:city_of_branch'],
                  36: ['x', 'b', 'per:countries_of_residence'], 37: ['x', 'g', 'per:date_of_birth'],
                  38: ['x', 'r', 'per:city_of_death']}

    fig, ax = plt.subplots()
    for g in np.unique(test_labels):
        #if g < 10:
        ix = np.where(test_labels == g)
        ax.scatter(test_tsne[ix, 0], test_tsne[ix, 1], marker=label_dict[g][0], c=label_dict[g][1], s=5)

    ax.legend()
    plt.show()
    plt.savefig('tsne_mix/' + file_name)
    pass


def GMM(X, k):
    print('performing GMM clustering ...')
    seed = 0
    gmm = GaussianMixture(n_components=k, tol=0.1, covariance_type='diag', max_iter=2000,
                          random_state=seed
                          ).fit(X)
    print("Fit finished.")
    pred = gmm.predict(X)

    centroids = gmm.means_

    print("predict_proba:")
    probabilities = gmm.predict_proba(X)
    print(probabilities)

    """
    print("log_likelihoods:")
    log_likelihoods = gmm.score_samples(X)
    print(log_likelihoods)
    """
    return pred, centroids, probabilities


def computePurity(gt_labels):
    unique, counts = np.unique(gt_labels, return_counts=True)
    index = np.argmax(counts)
    majority = unique[index]
    return majority, counts[index]

# find the k most possible train label candidates
def findTopkwsim(vector, total_rel_num, k, gt_label):
    all_dict = {}
    gt_sim = 0
    #offset_gt_label = gt_label - int(gt_label/7) - 1
    offset_gt_label = gt_label
    for i in range(total_rel_num):
        cur_S_vector = [0 for j in range(total_rel_num)]
        cur_S_vector[i] = 1
        cur_sim = 1 - spatial.distance.cosine(cur_S_vector, vector)
        all_dict[i] = cur_sim
        if i == offset_gt_label:
            gt_sim = cur_sim
    sorted_dict = dict(sorted(all_dict.items(), key=lambda item: item[1], reverse=True))
    cnt_k = 0
    k_dict = {}
    for key in sorted_dict:
        if cnt_k < k:
            # add the label offset back
            #offset = int(key/6) + 1
            offset = 0
            k_dict[str(key+offset)] = float('{:.5f}'.format(sorted_dict[key]))
            cnt_k += 1
    return k_dict, gt_sim


def countRecall(dict, gt_label):
    r1 = 0
    r3 = 0
    r5 = 0
    cnt = 0
    for key in dict:
        if key == str(gt_label):
            if cnt < 1:
                r1, r3, r5 = 1, 1, 1
            elif cnt < 3:
                r3, r5 = 1, 1
            else:
                r5 = 1
        cnt += 1
    return r1, r3, r5



# find the most possible train label candidate
def findCLabel(vector):
    label = -1
    min_gap = 1
    for idx in range(len(vector)):
        if abs(1-vector[idx]) < min_gap:
            min_gap = abs(1-vector[idx])
            label = idx
    return label


def computeGTsim(vector, total_rel_num, gt_label):
    cur_S_vector = [0 for j in range(total_rel_num)]
    cur_S_vector[gt_label] = 1
    gt_sim = 1 - spatial.distance.cosine(cur_S_vector, vector)
    return gt_sim



if __name__ == '__main__':
    main()
    pass

"""

        threshold_lst = []
        num_remain_lst = []
        wcp_lst = []
        threshold = 1.00
        while threshold > 0:
            num_remain = 0
            o_remain_lst = []
            o_remain_label_lst = []
            for i in range(len(d)):
                ori_dict, cur_ori_gt_sim = findTopkwsim(d[i], top_rel_num, 5, label[i])
                ori_fst_sim = 0
                ori_key = 0
                rw_fst_sim = 0
                rw_key = 0
                for key in ori_dict:
                    ori_fst_sim = ori_dict[key]
                    ori_key = key
                    break

                # belongs to remain_intersection
                if ori_fst_sim < threshold and rw_fst_sim < threshold:
                    num_remain += 1
                    o_remain_lst.append(d[i])
                    o_remain_label_lst.append(label[i])

            num_clusters = len(novel_label_lst)
            if num_remain < num_clusters:
                break

            pred_labels, mean_vectors, pred_probabilities = GMM(o_remain_lst, num_clusters)
            num_purity_lst = []
            for c in range(num_clusters):
                tmp_label_lst = []
                for j in range(len(o_remain_lst)):
                    if pred_labels[j] == c:
                        tmp_label_lst.append(o_remain_label_lst[j])
                if len(tmp_label_lst) == 0:
                    continue
                majority_label, cnt = computePurity(tmp_label_lst)
                purity = '{:.5f}'.format(cnt/len(tmp_label_lst))
                num_purity_lst.append([len(tmp_label_lst), float(purity)])

            wcp = 0
            for p in num_purity_lst:
                wcp += p[0] * p[1] / (len(o_remain_lst))
            wcp = '{:.3f}'.format(wcp)
            threshold_lst.append(threshold)
            num_remain_lst.append(num_remain)
            wcp_lst.append(float(wcp))
            threshold -= 0.05

        print(threshold_lst)
        print(num_remain_lst)
        print(wcp_lst)
        p_lst = [int(x*y) for x, y in zip(num_remain_lst, wcp_lst)]
        print(p_lst)
        print()

        plt.subplot(1, 2, 1)
        plt.title('FewRel')
        plt.plot(threshold_lst, num_remain_lst, 's-', color='r')
        plt.xlabel('threshold')
        plt.ylabel('num_remain')

        plt.subplot(1, 2, 2)
        plt.plot(threshold_lst, wcp_lst, 'o-', color='g')
        plt.xlabel('threshold')
        plt.ylabel('weighted_cluster_purity')

        plt.show()


"""

"""
a = [17500, 17363, 16444, 15171, 13856, 12718, 11433, 10033, 8353, 6356, 4096, 2078, 702, 144, 8]
b = [0.363, 0.358, 0.352, 0.34, 0.341, 0.359, 0.366, 0.393, 0.418, 0.443, 0.464, 0.538, 0.613, 0.618, 0.875]
c =[]
for i in range(len(a)):
    c1 = a[i] * b[i]
    c.append(int(c1))
print(c)
"""
