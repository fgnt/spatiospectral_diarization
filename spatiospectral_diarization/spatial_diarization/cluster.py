import numpy as np


def pseudo_hamming_distance(tdoas, ref_tdoas, margin=5):
    dist = np.sum(np.abs(tdoas - ref_tdoas) > margin)
    return dist

def temporally_constrained_clustering(candidates, max_dist=5, max_temp_dist=16, peak_ratio_th=.75):
    max_peaks = []
    for c in candidates:
        if len(c):
            if len(c[-1]) == 1:
                max_peaks.append(c[-1][0][-1])
    srp_th = 0#np.mean(max_peaks) - 2 * np.std(max_peaks)
    diary = []
    for (frame_id, spk_pos) in candidates:
        if not len(spk_pos):
            continue
        (tdoas, srp) = spk_pos[0]
        if srp < srp_th:
            continue
        tdoas = np.asarray(tdoas)
        if not len(diary):
            diary.append(([tdoas], [frame_id], [srp]))
            continue
        else:
            for c, entry in enumerate(diary):
                ref_tdoas = entry[0][-1]
                if frame_id - entry[1][-1] > max_temp_dist:
                    continue
                if np.linalg.norm(ref_tdoas - tdoas) <= max_dist and np.max(np.abs(ref_tdoas - tdoas)) <= .5:
                    entry[0].append(tdoas)
                    entry[1].append(frame_id)
                    entry[2].append(srp)
                    break
            else:
                diary.append(([tdoas], [frame_id], [srp]))
        ref_srp = srp
        if ref_srp < srp_th:
            continue
        for (tdoas, srp) in spk_pos[1:]:
            if srp / ref_srp < peak_ratio_th:
                continue
            if srp < srp_th:
                continue
            tdoas = np.asarray(tdoas)
            if not len(diary):
                diary.append(([tdoas], [frame_id], [srp], [tdoas]))
                continue
            else:
                for c, entry in enumerate(diary):
                    if frame_id - entry[1][-1] > max_temp_dist:
                        continue
                    match = False
                    for x in range(len(entry[0])):
                        if frame_id - entry[1][x] > max_temp_dist:
                            continue
                        if np.linalg.norm(entry[0][x] - tdoas) <= max_dist and np.max(np.abs(entry[0][x] - tdoas)) <= .5:
                            entry[0].append(tdoas)
                            entry[1].append(frame_id)
                            entry[2].append(srp)
                            match = True
                            break
                    if match:
                        break
                else:
                    diary.append(([tdoas], [frame_id], [srp]))
    diary = sorted(diary, key=lambda x: x[1][0], reverse=True)
    return diary

def single_linkage_clustering(temp_diary, max_dist=2):
    M = np.zeros((len(temp_diary), len(temp_diary)))

    for i in range(len(temp_diary)):
        for j in range(i + 1, len(temp_diary)):
            M[i, j] = M[j, i] = np.max(
                np.mean(
                    (np.median(temp_diary[i][0], 0)
                     - np.median(temp_diary[j][0], 0)) ** 2
                )
            )
    op = np.minimum  # single-linkage clustering

    groups = [(i,) for i in range(M.shape[0])]

    indices = np.argsort(M.ravel())
    indices = zip(indices // M.shape[0], indices % M.shape[0])

    delete = []
    for i, (idx1, idx2) in enumerate(indices):
        if idx1 == idx2 or idx1 in delete or idx2 in delete:
            continue
        if M[idx1, idx2] <= max_dist:
            M[:, idx1] = M[idx1, :] = op(M[:, idx1], M[:, idx2])
            groups[idx1] += groups[idx2]
            delete.append(idx2)

    for d in sorted(delete, reverse=True):
        del groups[d]
    M = np.delete(M, delete, axis=0)
    M = np.delete(M, delete, axis=1)
    return groups, M

