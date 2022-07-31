import numpy as np
from hmmlearn import hmm

def distance(x1, x2) :
    return np.sqrt(np.sum((x1-x2)**2))

def dtw(M1, M2) :
    '''Use DTW to calculate the similarity distance between two MFCC features. 

    M1: The first MFCC feature. 
    M2: The first MFCC feature. 

    return: A float object of the similarity distance between two MFCC features.
    '''
    M1_len = M1.shape[0]
    M2_len = M2.shape[0]
    cost = [[0 for i in range(M2_len)] for i in range(M1_len)]
    
    dis = []
    for i in range(M1_len) :
        dis_row = []
        for j in range(M2_len) :
            dis_row.append(distance(M1[i], M2[j]))
        dis.append(dis_row)
    
    cost[0][0] = dis[0][0]
    for i in range(1, M1_len) :
        cost[i][0] = cost[i - 1][0] + dis[i][0]
    for j in range(1, M2_len) :
        cost[0][j] = cost[0][j - 1] + dis[0][j]
    
    for i in range(1, M1_len) :
        for j in range(1, M2_len) :
            cost[i][j] = min(cost[i - 1][j] + dis[i][j] * 1, \
                            cost[i- 1][j - 1] + dis[i][j] * 2, \
                            cost[i][j - 1] + dis[i][j] * 1)
    return cost[M1_len - 1][M2_len - 1]

class GMMHMM:
    def __init__(self, features, labels, n_iter = 10):
        '''Construct and train a GMM+HMM model. 

        features: A list consisting of MFCC features.
        labels: The label corresponding to each MFCC feature in the features list. 
        n_iter: Iterating times. 

        return: A GMMHMM object. 
        '''
        GMMHMM_Models = {}
        states_num = 5
        GMM_mix_num = 6
        tmp_p = 1.0/(states_num-2)
        transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0 ,0], [0, tmp_p, tmp_p, tmp_p , 0], [0, 0, tmp_p, tmp_p,tmp_p], [0, 0, 0, 0.5, 0.5], [0, 0, 0, 0, 1]],dtype=np.float)
        startprobPrior = np.array([0.5, 0.5, 0, 0, 0],dtype=np.float) 
        for label in np.unique(labels):
            trainData = [feature for feature, l in zip(features, labels) if l == label]
            model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, transmat_prior=transmatPrior, startprob_prior=startprobPrior, covariance_type='diag', n_iter=n_iter) 
            length = np.zeros([len(trainData), ], dtype=np.int) 
            for m in range(len(trainData)):
                length[m] = trainData[m].shape[0]
            trainData = np.vstack(trainData)
            model.fit(trainData, lengths=length)
            GMMHMM_Models[label] = model 
        self.GMMHMM_Models = GMMHMM_Models
    
    def predict(self, features):
        '''Use the trained GMM+HMM model to predict the labels on test set. 

        features: A list consisting of MFCC features.

        return: A list of predicted labels. 
        '''
        labels = []
        for feature in features:
            scoreList = {}
            for model_label in self.GMMHMM_Models.keys():
                model = self.GMMHMM_Models[model_label]
            score = model.score(feature)
            scoreList[model_label] = score
            predict = max(scoreList, key=scoreList.get)  
            labels.append(predict)
        return labels