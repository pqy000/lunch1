# metrics.py

def compute_faa_aaa_wca(acc_matrix):
    """
    acc_matrix[t][i]: after training task t, test on task i => accuracy in [0..1]
    """
    T = len(acc_matrix)
    # FAA
    final_row = acc_matrix[T-1]
    FAA = sum(final_row)/T

    # AAA
    AAA = 0.0
    for j in range(T):
        AAA += sum(acc_matrix[j][:j+1])/(j+1)
    AAA /= T

    # WCA
    if T>1:
        minvals=[]
        for i in range(T-1):
            local_min = min(acc_matrix[j][i] for j in range(i+1, T))
            minvals.append(local_min)
        minAcc_T = sum(minvals)/len(minvals) if len(minvals)>0 else 0.0
    else:
        minAcc_T= 0.0
    final_acc = acc_matrix[T-1][T-1]
    WCA = (1.0/T)*final_acc + (1.0 - 1.0/T)*minAcc_T
    return FAA, AAA, WCA