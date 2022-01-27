import AirPort_lib as mdl
import pandas as pd

def getCSV(file):
    return pd.read_csv(file, engine='python')

#2017年データの取り込み
df_data = getCSV('./csv/Matsumoto_modify02.csv')
#print(df_data.describe())
#print(df_data.shape)

n_components = 0.8 #PCAパラメタ
alib = mdl.AirPort_lib(df_data, n_components)
alib.getPCA() #PCAの実行
labels = alib.getKmeans(6,111) #kmeansの実行

alib.getHistgram(df_data, labels)
alib.getTtest(df_data, labels)

#2019年データの取り込み
#新規データの取り込み
df_test = getCSV('./csv/Matsumoto2019.csv')
#print(df_test.describe())
#検証データをpca行列で変換
feature_test = alib.transPCA(df_test) 
print(feature_test.shape)

#SVMで危険クラスタとそうでないものを利用して分類していく
svc_predict_test = alib.getSVM(labels, feature_test)
alib.getHistgram(df_test, svc_predict_test)
print(svc_predict_test)
sum = len(svc_predict_test)
for i in svc_predict_test:
    sum -= i
rate = sum / len(svc_predict_test) * 100
print('危険日 = %.2f' % sum)
print('危険度 = %.2f' % rate)

#20220128 テストデータでのリスククラスタと他のクラスタの２群の平均値の比較
print('Test Data')
alib.getTtest(df_test, svc_predict_test)


#20220125 SVMでの検証(リスククラスタを使わない場合)
#PCA->SVMでの教師モデルでテストデータの分類を実施した場合、すべて安全とみなされた。
print('')
print('20220125追加項目：SVMでの検証')
alib.getSVM2(feature_test)