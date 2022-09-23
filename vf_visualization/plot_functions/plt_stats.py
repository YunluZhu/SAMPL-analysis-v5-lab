import numpy as np

def calc_ROC(data,feature,ctrl_name,change_dir):
    '''
    data: long format dataframe containing feature to calculate and both cond and ctrl data
    feature: col name of the col in data to plot
    ctrl_name: name of the control in the condition column
    change_dir: expected direction of change in cond data, left or right
    '''
    ctrl_all = data.loc[data.condition==ctrl_name,feature].values
    cond_all = data.loc[data.condition!=ctrl_name,feature].values
    auc = []
    ymin = data[feature].min()
    ymax = data[feature].max()
    num = len(ctrl_all)-1
    interval = (ymax-ymin)/num
    x = np.linspace(ymin-interval,ymax,num=num)
    if change_dir == 'right':
        x = np.linspace(ymax+interval,ymin,num=num)
    # sensitivity
    for jack in range(len(ctrl_all)-1):
        ctrl = np.delete(ctrl_all,jack)
        cond = np.delete(cond_all,jack)
        total_ctrl = np.sum(ctrl)
        total_cond = np.sum(cond)
        TPR_list=[]
        FPR_list=[]
        for i,x_val in enumerate(x):
            if change_dir == 'right':
                cum_TP = cond[cond>=x_val].sum()
                cum_FP = ctrl[ctrl>=x_val].sum()
            else:
                cum_TP = cond[cond<=x_val].sum()
                cum_FP = ctrl[ctrl<=x_val].sum()
            FPR=cum_FP/total_ctrl
            TPR=cum_TP/total_cond
            TPR_list.append(TPR)
            FPR_list.append(FPR)
        this_auc=np.sum(TPR_list)/len(ctrl)
        # this_ROC = pd.DataFrame(data={
        #     'FPR':FPR_list,
        #     'TPR':TPR_list,
        #     'jackknife_num':[jack]*len(x),
        # })
        # ROC = pd.concat([ROC,this_ROC],ignore_index=True)
        auc.append(this_auc)

    # repeat with no jackknife for the curve
    ymin = data[feature].min()
    ymax = data[feature].max()
    num = len(ctrl_all)
    interval = (ymax-ymin)/num
    x = np.linspace(ymin-interval,ymax,num=num)
    if change_dir == 'right':
        x = np.linspace(ymax+interval,ymin,num=num)
    total_ctrl = np.sum(ctrl_all)
    total_cond = np.sum(cond_all)
    TPR_list=[]
    FPR_list=[]
    for i,x_val in enumerate(x):
        if change_dir == 'right':
            cum_TP = cond_all[cond_all>=x_val].sum()
            cum_FP = ctrl_all[ctrl_all>=x_val].sum()
        else:
            cum_TP = cond_all[cond_all<=x_val].sum()
            cum_FP = ctrl_all[ctrl_all<=x_val].sum()
        FPR=cum_FP/total_ctrl
        TPR=cum_TP/total_cond
        TPR_list.append(TPR)
        FPR_list.append(FPR)
        
    return TPR_list, FPR_list, auc