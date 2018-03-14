from train import *

acc_result = np.load('feature_hr.npy')

print(acc_result.T[0][0])
print(acc_result.T[1][0])
print(acc_result.T[2][0])
# plt.plot(
#     acc_result.T[0][0],
#     acc_result.T[1][0],
#     acc_result.T[0][0],
#     acc_result.T[2][0]
# )
# for i,j in zip(acc_result.T[0][0],acc_result.T[1][0]):
#     plt.text(i,j,str(j))
# for i,j in zip(acc_result.T[0][0],acc_result.T[2][0]):
#     plt.text(i, j, str(j))
# plt.legend(['Training','Validation'])
# plt.ylabel('ave cost')
# plt.xlabel('ignore term')
# plt.title('Compare between diff feature HRs selected')
# plt.savefig('feature_HR.png')

topic = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10',
         'PM2.5','RAINFALL','RH','SO2','THC','WD_HR','WIND_DIREC',
         'WIND_SPEED','WS_HR','PM2.5^2']

for i in range(len(topic)):
    #print(acc_result.T[1][0][0] + acc_result.T[1][0][9 * i + 1:9 * i + 11],list(range(-1,10)))
    plt.scatter(
        acc_result.T[0][0][10 * i:10 * i + 10],
        acc_result.T[1][0][10*i:10*i+10])
    plt.scatter(
        acc_result.T[0][0][10 * i:10 * i + 10],
        acc_result.T[2][0][10 * i:10 * i + 10])
    plt.legend(['Training','Validation'])
    plt.axhline(y=acc_result.T[1][0][10*i+9], color='blue', linestyle='-')
    plt.axhline(y=acc_result.T[2][0][10*i+9], color='orange', linestyle='-')
    plt.ylabel('ave cost')
    plt.xlabel('selected HRs')
    plt.title('Compare between diff '+topic[i]+' selected HRs')
    plt.savefig('feature_plot\\feature_'+topic[i]+'_selected_HRs.png')
    plt.clf()
