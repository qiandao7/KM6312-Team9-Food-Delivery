import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

base=Path(__file__).resolve().parent
out=base/'report_figures'
out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'text.color':'black','axes.labelcolor':'black','axes.edgecolor':'black','xtick.color':'black','ytick.color':'black','axes.linewidth':0.6,'lines.linewidth':0.9,'savefig.dpi':600,'figure.facecolor':'white','axes.facecolor':'white','legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False})
data=pd.read_csv(base/'food_delivery_analytics_cleaned.csv')
main=data[data.cancellation_flag==False].copy()
train,test=train_test_split(main,test_size=.2,random_state=42)
features=['city_tier','order_hour','order_day_of_week','order_month','delivery_distance_km','preparation_time_minutes','traffic_level_score','weather_severity_score','delivery_partner_experience_years','number_of_items','festival_or_weekend_flag']
targets=['delivery_time_minutes','delivery_efficiency_score']
cutoff=train.delivery_efficiency_score.quantile(.25)
BLUE='#4477AA'
TEAL='#287C8E'
ORANGE='#E58B36'
PURPLE='#8B68B6'
GREEN='#4F9A6C'
RED='#CB5A59'
MODEL_COLORS={'Logistic':TEAL,'Random forest':PURPLE,'Gradient boosting':ORANGE}
manifest=[]
def save(fig,name):
    for artist in fig.findobj(matplotlib.text.Text): artist.set_color('black')
    fig.savefig(out/name,dpi=600,facecolor='white')
    manifest.append({'file':name,'width_in':float(fig.get_figwidth()),'height_in':float(fig.get_figheight())})
    plt.close(fig)

fig,ax=plt.subplots(figsize=(3.25,2.35));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,h,text,fill='#D8E9F5',edge=BLUE):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.01,rounding_size=0.015',facecolor=fill,edgecolor=edge,linewidth=.7))
    ax.text(x,y,text,ha='center',va='center',fontsize=9)
def arrow(a,b): ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':BLUE,'lw':.7})
box(.5,.9,.89,.14,'15,000 synthetic delivery records')
box(.5,.66,.89,.14,'12,997 non-cancelled orders')
box(.5,.43,.89,.15,'80/20 split: 10,397 train / 2,600 test')
arrow((.5,.82),(.5,.74));arrow((.5,.58),(.5,.52))
for x,text,fill,edge in [(.16,'RQ1\nAssociations\nTraining only','#DCEEDD',GREEN),(.5,'RQ2\nDelivery-time\nprediction','#E8DFF1',PURPLE),(.84,'RQ3\nLow-efficiency\nclassification','#FBE4CB',ORANGE)]:
    box(x,.13,.29,.22,text,fill,edge);arrow((x,.35),(x,.25))
fig.subplots_adjust(left=.015,right=.985,top=.98,bottom=.02);save(fig,'Figure01_Workflow.png')

fig,axs=plt.subplots(2,1,figsize=(3.25,2.8))
for ax,col,label in zip(axs,targets,['Delivery time (minutes)','Efficiency score']):
    ax.hist(train[col],bins=30,color=BLUE if col=='delivery_time_minutes' else TEAL,edgecolor='white',linewidth=.35);ax.set(xlabel=label,ylabel='Orders')
axs[1].axvline(cutoff,color=ORANGE,linestyle='--',linewidth=1)
axs[1].text(.03,.89,f'Cutoff = {cutoff}',transform=axs[1].transAxes,fontsize=9)
fig.tight_layout(pad=.8,h_pad=.9);save(fig,'Figure02_Distributions.png')

factors=['delivery_distance_km','preparation_time_minutes','traffic_level_score','weather_severity_score','delivery_partner_experience_years','number_of_items','festival_or_weekend_flag']
corr=train[factors+targets].corr().loc[factors,targets]
fig,ax=plt.subplots(figsize=(3.25,2.45))
cmap=LinearSegmentedColormap.from_list('notebook_red_blue',plt.get_cmap('RdBu_r')(np.linspace(.18,.82,256)))
ax.imshow(corr.to_numpy(),cmap=cmap,vmin=-1,vmax=1,aspect='auto')
ax.set_xticks([0,1],['Delivery time','Efficiency score']);ax.set_yticks(range(7),['Distance','Preparation time','Traffic','Weather','Experience','Item count','Festival/weekend'])
ax.tick_params(length=0)
for i in range(7):
    for j in range(2): ax.text(j,i,f'{corr.iloc[i,j]:.3f}',ha='center',va='center',fontsize=9)
for s in ax.spines.values():s.set_visible(False)
fig.subplots_adjust(left=.38,right=.98,top=.98,bottom=.12);save(fig,'Figure03_Correlations.png')

fig,axs=plt.subplots(2,1,figsize=(3.25,3.4),sharey=True)
for ax,col,label in zip(axs,['delivery_distance_km','preparation_time_minutes'],['Distance (km)','Preparation time (minutes)']):
    ax.scatter(train[col],train.delivery_time_minutes,s=2.2,alpha=.16,c=BLUE if col=='delivery_distance_km' else TEAL,linewidths=0,rasterized=True)
    ax.set(xlabel=label,ylabel='Delivery time (min)')
fig.tight_layout(pad=.8,h_pad=.9);save(fig,'Figure04_Operational_Scatter.png')

fig,axs=plt.subplots(2,1,figsize=(3.25,2.9))
for col,label,sty,marker in [('traffic_level_score','Traffic','-','o'),('weather_severity_score','Weather','--','s')]:
    groups=pd.qcut(train[col],4,labels=['Q1','Q2','Q3','Q4'])
    means=train.groupby(groups,observed=True)[targets].mean()
    for i,target in enumerate(targets):axs[i].plot(range(1,5),means[target],linestyle=sty,marker=marker,markersize=3.5,color=ORANGE if label=='Traffic' else PURPLE,label=label)
axs[0].set(ylabel='Mean delivery time (min)');axs[1].set(ylabel='Mean efficiency score',xlabel='Within-feature quartile: low to high')
for ax in axs:ax.set_xticks(range(1,5),['Q1','Q2','Q3','Q4'])
axs[0].legend(loc='upper left');fig.tight_layout(pad=.8,h_pad=.6);save(fig,'Figure05_Environment.png')

exp=pd.cut(train.delivery_partner_experience_years,bins=[0,4,8,12,15],labels=['1-4','5-8','9-12','13-15'])
fig,axs=plt.subplots(2,1,figsize=(3.25,2.9))
for ax,target,ylabel in zip(axs,targets,['Delivery time (minutes)','Efficiency score']):
    samples=[train.loc[exp==g,target] for g in exp.cat.categories]
    boxes=ax.boxplot(samples,tick_labels=list(exp.cat.categories),patch_artist=True,widths=.55,boxprops={'facecolor':'#BFD6E6' if target=='delivery_time_minutes' else '#BBDDD5','edgecolor':'black','linewidth':.6},medianprops={'color':'black','linewidth':.9},whiskerprops={'color':'black','linewidth':.6},capprops={'color':'black','linewidth':.6},flierprops={'marker':'.','markerfacecolor':'black','markeredgecolor':'black','markersize':1.5})
    for patch,shade in zip(boxes['boxes'],['#D9EEDC','#B5DDBD','#8BCB9E','#66B887']):patch.set_facecolor(shade)
    ax.set(ylabel=ylabel)
axs[1].set_xlabel('Partner experience (years)');fig.tight_layout(pad=.8,h_pad=.6);save(fig,'Figure06_Experience.png')

reg=pd.read_csv(base/'outputs/step04/RQ2_Test_Predictions.csv')
fig,axs=plt.subplots(2,1,figsize=(3.25,3.45),sharex=True,sharey=True)
for ax,name in zip(axs,['Ridge','Existing ETA']):
    ax.scatter(reg.actual_delivery_time,reg[name],s=3,alpha=.24,color=BLUE if name=='Ridge' else ORANGE,linewidths=0)
    ax.plot([0,180],[0,180],color='black',linestyle='--',lw=.8)
    ax.text(.04,.86,name,transform=ax.transAxes,fontsize=9)
    ax.set(ylabel='Predicted time (min)',xlim=(0,185),ylim=(0,185))
axs[1].set_xlabel('Actual time (minutes)');fig.tight_layout(pad=.8,h_pad=.6);save(fig,'Figure07_Actual_Predicted.png')

errors=reg.actual_delivery_time-reg.Ridge
fig,axs=plt.subplots(2,1,figsize=(3.25,3.2))
axs[0].hist(errors,bins=30,color=TEAL,edgecolor='black',linewidth=.35);axs[0].axvline(0,color='black',linestyle='--');axs[0].set(xlabel='Actual minus predicted (minutes)',ylabel='Orders')
axs[1].scatter(reg.Ridge,errors,s=3,alpha=.24,c=BLUE,linewidths=0);axs[1].axhline(0,color='black',linestyle='--');axs[1].set(xlabel='Predicted time (minutes)',ylabel='Error (minutes)')
fig.tight_layout(pad=.8,h_pad=.8);save(fig,'Figure08_Error_Diagnostics.png')

pred=pd.read_csv(base/'outputs/step05/RQ3_Test_Predictions.csv')
cm=confusion_matrix(pred.true_low_efficiency,pred.Logistic_predicted)
fig,ax=plt.subplots(figsize=(3.25,2.65));ax.imshow(cm,cmap=LinearSegmentedColormap.from_list('light_blue',['#F1F6FB','#6CA6D0']),aspect='auto',vmin=0,vmax=cm.max())
for i in range(2):
    for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=11)
ax.set_xticks([0,1],['Other','Low efficiency']);ax.set_yticks([0,1],['Other','Low efficiency']);ax.set(xlabel='Predicted class',ylabel='Actual class');ax.tick_params(length=0)
for s in ax.spines.values():s.set_visible(False)
fig.tight_layout(pad=.9);save(fig,'Figure09_Confusion_Matrix.png')

fig,ax=plt.subplots(figsize=(3.25,2.65))
for name,sty in [('Logistic','-'),('Random forest','--'),('Gradient boosting',':')]:
    precision,recall,_=precision_recall_curve(pred.true_low_efficiency,pred[name+'_probability'])
    ap=average_precision_score(pred.true_low_efficiency,pred[name+'_probability'])
    ax.plot(recall,precision,color=MODEL_COLORS[name],linestyle=sty,label=f'{name} ({ap:.3f})')
ax.axhline(pred.true_low_efficiency.mean(),color='.4',linestyle='-.',label='Positive-class share')
ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1.02),ylim=(0,1.02));ax.legend(loc='lower left',fontsize=8.5,frameon=True,facecolor='white',edgecolor='white',framealpha=1)
fig.tight_layout(pad=.8);save(fig,'Figure10_Precision_Recall.png')

imp=pd.read_csv(base/'outputs/step05/RQ3_Permutation_Importance.csv')
labels={'city_tier':'City tier','order_hour':'Order hour','order_day_of_week':'Weekday code','order_month':'Month','delivery_distance_km':'Distance','preparation_time_minutes':'Preparation time','traffic_level_score':'Traffic','weather_severity_score':'Weather','delivery_partner_experience_years':'Experience','number_of_items':'Item count','festival_or_weekend_flag':'Festival/weekend'}
imp=imp.sort_values('AUC decrease',ascending=True)
fig,ax=plt.subplots(figsize=(3.25,2.85));ax.barh(imp.Feature.map(labels),imp['AUC decrease'],xerr=imp.SD,color=imp.Feature.map({'delivery_distance_km':BLUE,'preparation_time_minutes':TEAL,'traffic_level_score':ORANGE,'weather_severity_score':PURPLE,'delivery_partner_experience_years':GREEN}).fillna('#BCC4CE'),edgecolor='black',linewidth=.35,error_kw={'ecolor':'black','elinewidth':.65,'capsize':1.5});ax.set_xlabel('Decrease in test ROC-AUC');fig.subplots_adjust(left=.38,right=.97,bottom=.18,top=.98);save(fig,'Figure11_Importance.png')

categories=['city_tier','order_hour','order_day_of_week','order_month'];numeric=[c for c in features if c not in categories+['festival_or_weekend_flag']]
pre=ColumnTransformer([('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categories),('numeric',StandardScaler(),numeric),('weekend','passthrough',['festival_or_weekend_flag'])])
clf=make_pipeline(pre,LogisticRegression(C=1,max_iter=2000,random_state=42)).fit(train[features],(train.delivery_efficiency_score<=cutoff).astype(int))
assert np.allclose(clf.predict_proba(test[features])[:,1],pred.Logistic_probability)
pdp_features=['delivery_distance_km','traffic_level_score','weather_severity_score','delivery_partner_experience_years']
reference=train[features].sample(1000,random_state=42).astype({c:float for c in pdp_features})
fig,axs=plt.subplots(2,2,figsize=(3.25,3.4))
display=PartialDependenceDisplay.from_estimator(clf,reference,pdp_features,kind='both',subsample=25,grid_resolution=30,random_state=42,method='brute',response_method='predict_proba',n_jobs=1,ax=axs,pd_line_kw={'color':TEAL,'linewidth':1.2,'linestyle':'-'},ice_lines_kw={'color':'#7DA9CC','alpha':.3,'linewidth':.5})
for i,(ax,label) in enumerate(zip(axs.ravel(),['Distance (km)','Traffic score','Weather score','Experience (years)'])):
    ax.set(xlabel=label,ylabel='Probability' if i%2==0 else '',ylim=(0,1),yticks=[0,.5,1])
    if i%2:ax.set_yticklabels([])
    if ax.get_legend() is not None:ax.get_legend().remove()
fig.tight_layout(pad=.6,w_pad=.8,h_pad=.9);save(fig,'Figure12_PDP_ICE.png')

stage=pd.read_csv(base/'outputs/step06/Sensitivity_Prediction_Stage.csv')
fig,axs=plt.subplots(1,2,figsize=(3.25,2.05),sharey=True)
for ax,col,xlabel in zip(axs,['MAE','F1'],['MAE (minutes)','Classification F1']):
    ax.barh([0,1],stage[col],height=.5,color=[TEAL,RED],edgecolor='black',linewidth=.4)
    ax.set_xlabel(xlabel)
    for y,v in enumerate(stage[col]):ax.text(v+.02 if col=='F1' else v+.2,y,f'{v:.3f}',va='center',ha='left',fontsize=8.5)
axs[0].invert_yaxis();axs[0].set_yticks([0,1],['Prep known','Prep unknown']);axs[0].set_xlim(0,17);axs[1].set_xlim(0,1.2)
fig.tight_layout(pad=.7,w_pad=.6);save(fig,'Figure13_Information_Stage.png')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'figures':len(manifest),'all_chart_text_black':True,'classifier_predictions_match_saved_results':True,'manifest':str(out/'manifest.json')}))
