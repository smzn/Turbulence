import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import scipy #等分散性の検定(F件検定)を行うためのライブラリ

class AirPort_lib :
    def __init__(self, df_data, n_components):
        self.df_data = df_data #教師ラベル(mod_turbあり)
        self.t_label = df_data['mod_turb']
        #print(self.t_label)
        self.df_data_rm = df_data.drop(columns='mod_turb')
        self.df_data_sc = self.getSC(self.df_data_rm) #教師データ削除して正規化(これを利用)
        #print(self.df_data_sc.shape)
        self.n_components= n_components

    def saveCSV(self, df, fname):
        pdf = pd.DataFrame(df) #データフレームをpandasに変換
        pdf.to_csv(fname)

    def getSC(self, df):
        sc = StandardScaler()
        return sc.fit_transform(df)

    def getPCA(self):
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(self.df_data_sc)
        self.feature = self.pca.transform(self.df_data_sc) #featureはデータセットを変換した値(Z)
        #print(feature.shape)
        #print(feature)
        self.components = self.pca.components_ #componentsは変換用行列(W)
        #print(self.components.shape)
        self.explained_variance = self.pca.explained_variance_ #固有値
        #print(self.pca.explained_variance_) 
        self.explained_variance_ratio = self.pca.explained_variance_ratio_ #因子寄与率
        #print(self.explained_variance_ratio)
        self.saveCSV(self.feature, './csv/feature.csv')

    def transPCA(self, df_test): #検証データを主成分を使って変換
        #dfを正規化
        df_test_sc = self.getSC(df_test)
        feature_test = self.pca.transform(df_test_sc) #featureはデータセットを変換した値(Z)
        return feature_test

    def getKmeans(self, n_cltr, rdm_state):
        kmodel = KMeans(n_clusters= n_cltr, random_state= rdm_state)
        kmodel.fit(self.feature)
        self.kmodel = kmodel
        self.kmodel_labels = kmodel.labels_
        self.kmodel_cluster_centers = kmodel.cluster_centers_
        self.kmodel_inertia = kmodel.inertia_
        print(kmodel.labels_) #kmeansの結果ラベル
        #print(kmodel.labels_.shape)
        #print(kmodel.cluster_centers_) #クラスタの中心座標
        #print(kmodel.inertia_) #最も近いクラスタ中心までの距離の二乗和
        plt.scatter(self.feature[:,0],self.feature[:,1], c=kmodel.labels_)
        plt.scatter(kmodel.cluster_centers_[:,0], kmodel.cluster_centers_[:,1],s=250, marker='*',c='red')
        plt.grid()
        plt.show()
        return kmodel.labels_

    def getSVM(self, labels, feature_test):
        for i,val in enumerate(labels):
            if(val > 0):
                labels[i] = 1
        print(labels)
        #df_data['kmeans'] = labels
        #print(df_data['kmeans'])
        svc_model = SVC(gamma='auto')
        svc_model.fit(self.feature, labels)
        svc_predict = svc_model.predict(self.feature)
        print(svc_predict)

        ac_acore = accuracy_score(labels, svc_predict)
        print('正解率 = %.2f' % (ac_acore))

        cm = confusion_matrix(labels, svc_predict)
        print(cm)
        tn, fp, fn, tp = cm.flatten()
        print('TP: True Positive = %.2f' % tp) #実際のクラスが陽性で予測も陽性（正解）
        print('TN: True Negative = %.2f' % tn) #実際のクラスが陰性で予測も陰性（正解）
        print('FP: False Positive = %.2f' % fp) #実際のクラスは陰性で予測が陽性（不正解）
        print('FN: False Negative = %.2f' % fn) #実際のクラスは陽性で予測が陰性（不正解）

        print('正解率 = %.2f' % (ac_acore))
        print('精度:precision = %.2f' % (precision_score(labels, svc_predict)))
        print('再現率:recall = %.2f' % recall_score(labels, svc_predict))
        print('F1 = %.2f' % f1_score(labels, svc_predict))
        print(classification_report(labels, svc_predict))

        index = np.where(svc_predict > 0)
        #plt.scatter(self.feature[:, 0], self.feature[:, 1])
        #plt.scatter(self.feature[index, 0], self.feature[index, 1], c='r', label='outlair')

        #検証データでの分類
        svc_predict_test = svc_model.predict(feature_test)
        print(svc_predict_test)
        return svc_predict_test
    
    def getSVM2(self, feature_test):#20220125 SVM検証(リスククラスタを使わない場合)
        svc_model = SVC(gamma='auto')
        svc_model.fit(self.feature, self.t_label)
        svc_predict = svc_model.predict(self.feature)
        print(svc_predict)
        ac_acore = accuracy_score(self.t_label, svc_predict)
        print('正解率 = %.2f' % (ac_acore))

        cm = confusion_matrix(self.t_label, svc_predict)
        print(cm)
        tn, fp, fn, tp = cm.flatten()
        print('TP: True Positive = %.2f' % tp) #実際のクラスが陽性で予測も陽性（正解）
        print('TN: True Negative = %.2f' % tn) #実際のクラスが陰性で予測も陰性（正解）
        print('FP: False Positive = %.2f' % fp) #実際のクラスは陰性で予測が陽性（不正解）
        print('FN: False Negative = %.2f' % fn) #実際のクラスは陽性で予測が陰性（不正解）

        print('正解率 = %.2f' % (ac_acore))
        print('精度:precision = %.2f' % (precision_score(self.t_label, svc_predict)))
        print('再現率:recall = %.2f' % recall_score(self.t_label, svc_predict))
        print('F1 = %.2f' % f1_score(self.t_label, svc_predict))
        print(classification_report(self.t_label, svc_predict))
        
        #検証データでの分類
        svc_predict_test = svc_model.predict(feature_test)
        print(svc_predict_test)
        sum = len(svc_predict_test)
        for i in svc_predict_test:
            sum -= i
        rate = sum / len(svc_predict_test) * 100
        print('危険日 = %.2f' % sum)
        print('危険度 = %.2f' % rate)


    def getHistgram(self, df_data, labels):
        df_data['kmeans'] = labels #ラベルを追加
        self.saveCSV(df_data, './csv/df_data.csv')
        #kmeansの各クラスタ平均値
        print("実際のデータでのクラスタの平均値")
        print(df_data[df_data['kmeans']==0].mean())

        df_data0 = df_data[(df_data['kmeans'] == 0 )] #危険クラスタのみ抽出
        graph_list = ['fx106_03_500spd','WA_700_12_hum', 'fx106_00_500spd','TA_500_12_hum','vis1','vis2 ','vis3','fx502_00_trough','fx502_00_alt','dwp','relh','fx106_00_500shear','fx106_03_500shear']
        for i in graph_list:
            plt.hist(df_data[i],density=True, label="All")
            plt.hist(df_data0[i],density=True, label="Risk", alpha=0.3)
            plt.legend(loc="upper left", fontsize=13)  
            plt.title(i, fontsize=15) 
            plt.show()

    def getdf_data_sc(self):
        return self.df_data_sc
    
    def getTtest(self, df_data, labels):
        df_data['kmeans'] = labels #ラベルを追加
        df_data_risk = df_data[(df_data['kmeans'] == 0 )] #危険クラスタのみ抽出
        df_data_others = df_data[(df_data['kmeans'] != 0 )] #危険クラスタのみ抽出
        graph_list = ['fx106_03_500spd','WA_700_12_hum', 'MA_500_12_hum','fx106_03_500shear']
        # 等分散性の検定を行う。
        for i in graph_list:
            print(i)
            stat, p = scipy.stats.bartlett(df_data_risk[i], df_data_others[i])
            print('Bartlett p_value = {0}'.format(p))
            if p >= 0.05:
                stat, p = scipy.stats.ttest_ind(df_data_risk[i], df_data_others[i])
                print('T Test p_value = {0}'.format(p))
            elif p < 0.05:
                stat, p = scipy.stats.ttest_ind(df_data_risk[i], df_data_others[i], equal_var=False)
                print('Welch p_value = {0}'.format(p))