import re, json, time, warnings, math
warnings.filterwarnings('ignore')
from collections import defaultdict, Counter
import numpy as np
from artifact_tool import Blob, SpreadsheetFile
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score, brier_score_loss
from xgboost import XGBClassifier

SEED=42
# ---------- Load & QC ----------
wb=SpreadsheetFile.import_xlsx(Blob.load('/mnt/data/dataset_anxiety(2)_2.xlsx'))
sh=wb.worksheets.get_item('Form Responses 1')
vals=sh.get_range('A1:Q515').values
rows=vals[1:]

def norm(s): return re.sub(r'\s+',' ',str(s).strip().lower())
def age_num(x):
    if isinstance(x,(int,float)) and not isinstance(x,bool): return int(float(x))
    m=re.search(r'\d+', str(x)); return int(m.group()) if m else None
G=defaultdict(list)
for idx,row in enumerate(rows): G[norm(row[1])].append((idx,row))
remove=set()
for email,g in G.items():
    if len(g)>1:
        seen=[]
        for idx,row in g:
            core=(age_num(row[3]), norm(row[5]), norm(row[6]), norm(row[7]))
            if core in seen: remove.add(idx)
            else: seen.append(core)
clean=[row for i,row in enumerate(rows) if i not in remove]
assert len(clean)==510

func_map={
    'tidak sulit sama sekali':0,
    'agak sulit':1,
    'sangat sulit':2,
    'sangat mengganggu aktivitas sehari-hari':3,
}
# M1 raw columns: age, sex, edu, employment, prevdiag(binary), func(ord)
X=[]; y=[]
for row in clean:
    score=sum(int(float(row[j])) for j in range(9,16))
    y.append(1 if score>=10 else 0)
    X.append([
        float(age_num(row[3])),
        str(row[5]).strip(),
        str(row[6]).strip(),
        str(row[7]).strip(),
        1.0 if norm(row[8])=='ya' else 0.0,
        float(func_map[norm(row[16])]),
    ])
X=np.array(X,dtype=object); y=np.asarray(y,dtype=int)

class BalancedXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_child_weight=1,
                 subsample=1.0, colsample_bytree=1.0, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0,
                 tree_method='hist', random_state=42):
        self.n_estimators=n_estimators; self.learning_rate=learning_rate; self.max_depth=max_depth
        self.min_child_weight=min_child_weight; self.subsample=subsample; self.colsample_bytree=colsample_bytree
        self.gamma=gamma; self.reg_alpha=reg_alpha; self.reg_lambda=reg_lambda
        self.tree_method=tree_method; self.random_state=random_state
    def fit(self,X,y):
        y=np.asarray(y)
        neg=(y==0).sum(); pos=(y==1).sum(); ratio=float(neg/pos) if pos else 1.0
        self.model_=XGBClassifier(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth,
            min_child_weight=self.min_child_weight, subsample=self.subsample, colsample_bytree=self.colsample_bytree,
            gamma=self.gamma, reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda,
            scale_pos_weight=ratio, tree_method=self.tree_method, random_state=self.random_state,
            objective='binary:logistic', eval_metric='logloss', n_jobs=1, verbosity=0)
        self.model_.fit(X,y)
        self.classes_=self.model_.classes_
        return self
    def predict_proba(self,X): return self.model_.predict_proba(X)
    def predict(self,X): return self.model_.predict(X)

# Preprocessing by model/config

def make_preproc(config, scaled_age):
    # raw indices for M0 [0..4], M1 [0..5]
    transformers=[]
    transformers.append(('age', StandardScaler() if scaled_age else 'passthrough', [0]))
    transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [1,2,3]))
    transformers.append(('diag','passthrough',[4]))
    if config=='M1': transformers.append(('func','passthrough',[5]))
    return ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)

def model_spec(name, config):
    scaled = name in ('LR','SVM')
    prep=make_preproc(config, scaled)
    if name=='LR':
        est=LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=5000, random_state=SEED)
        pipe=Pipeline([('prep',prep),('model',est)])
        params={'model__penalty':['l1','l2'],'model__C':[0.01,0.1,1,10,100]}
        return pipe, GridSearchCV, params, {}
    if name=='SVM':
        est=SVC(class_weight='balanced', probability=True, random_state=SEED)
        pipe=Pipeline([('prep',prep),('model',est)])
        params=[
            {'model__kernel':['linear'],'model__C':[0.1,1,10,100]},
            {'model__kernel':['rbf'],'model__C':[0.1,1,10,100],'model__gamma':['scale',0.001,0.01,0.1,1]},
        ]
        return pipe, GridSearchCV, params, {}
    if name=='RF':
        est=RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=1)
        pipe=Pipeline([('prep',prep),('model',est)])
        params={
            'model__n_estimators':[200,500], 'model__max_depth':[None,5,10],
            'model__min_samples_split':[2,5], 'model__min_samples_leaf':[1,2,4],
            'model__max_features':['sqrt','log2']}
        return pipe, GridSearchCV, params, {}
    if name=='XGBoost':
        est=BalancedXGBClassifier(random_state=SEED)
        pipe=Pipeline([('prep',prep),('model',est)])
        params={
            'model__n_estimators':[100,300,500], 'model__learning_rate':[0.01,0.05,0.10],
            'model__max_depth':[2,3,5], 'model__min_child_weight':[1,3,5],
            'model__subsample':[0.70,0.90,1.00], 'model__colsample_bytree':[0.70,0.90,1.00],
            'model__gamma':[0,0.10,0.50], 'model__reg_alpha':[0,0.10,1.00],
            'model__reg_lambda':[1,5,10]}
        return pipe, RandomizedSearchCV, params, {'n_iter':40,'random_state':SEED}
    raise ValueError(name)

outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
outer_splits=list(outer.split(X,y))
models=['LR','SVM','RF','XGBoost']
results={}
start=time.time()
for config in ['M0','M1']:
    Xc=X[:,:5] if config=='M0' else X
    for name in models:
        print(f'START {config} {name}',flush=True)
        oof=np.full(len(y),np.nan,float)
        fold_info=[]
        for fold,(tr,va) in enumerate(outer_splits,1):
            inner=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
            pipe, Search, params, extra=model_spec(name,config)
            search=Search(pipe, params, scoring='roc_auc', cv=inner, n_jobs=-1, refit=True, error_score='raise', **extra)
            t=time.time(); search.fit(Xc[tr],y[tr]);
            p=search.predict_proba(Xc[va])[:,1]; oof[va]=p
            fold_auc=roc_auc_score(y[va],p)
            fold_info.append({'fold':fold,'auc':float(fold_auc),'best_score_inner':float(search.best_score_),'best_params':search.best_params_})
            print(f'  fold {fold} auc={fold_auc:.4f} time={time.time()-t:.1f}s',flush=True)
        assert not np.isnan(oof).any()
        results[f'{config}_{name}']={'oof':oof.tolist(),'folds':fold_info}
        print(f'DONE {config} {name} auc={roc_auc_score(y,oof):.4f} elapsed={time.time()-start:.1f}s',flush=True)

# ---------- Metrics ----------
def cal_intercept_slope(y,p):
    eps=1e-6; p=np.clip(np.asarray(p,float),eps,1-eps); z=np.log(p/(1-p)).reshape(-1,1)
    # near-unpenalized calibration model
    m=LogisticRegression(C=1e6,solver='lbfgs',max_iter=10000)
    m.fit(z,y)
    return float(m.intercept_[0]), float(m.coef_[0,0])

def metric_set(y,p):
    pred=(np.asarray(p)>=0.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    ci,cs=cal_intercept_slope(y,p)
    return {
      'AUROC':float(roc_auc_score(y,p)), 'AUPRC':float(average_precision_score(y,p)),
      'Sensitivity':float(tp/(tp+fn)), 'Specificity':float(tn/(tn+fp)),
      'Precision':float(precision_score(y,pred,zero_division=0)), 'F1':float(f1_score(y,pred,zero_division=0)),
      'BalancedAccuracy':float(balanced_accuracy_score(y,pred)), 'Brier':float(brier_score_loss(y,p)),
      'CalibrationIntercept':ci, 'CalibrationSlope':cs,
      'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}
summary={k:metric_set(y,np.array(v['oof'])) for k,v in results.items()}

# Bootstrap 95% CI for key metrics; paired deltas M1-M0 by model
rng=np.random.default_rng(SEED); B=2000
boot_ci={}; deltas={}
for k,v in results.items():
    p=np.array(v['oof'])
    boots={m:[] for m in ['AUROC','AUPRC','Brier']}
    for b in range(B):
        idx=rng.integers(0,len(y),len(y)); yy=y[idx]
        if len(np.unique(yy))<2: continue
        pp=p[idx]
        boots['AUROC'].append(roc_auc_score(yy,pp)); boots['AUPRC'].append(average_precision_score(yy,pp)); boots['Brier'].append(brier_score_loss(yy,pp))
    boot_ci[k]={m:[float(np.quantile(a,.025)),float(np.quantile(a,.975))] for m,a in boots.items()}
for name in models:
    p0=np.array(results[f'M0_{name}']['oof']); p1=np.array(results[f'M1_{name}']['oof'])
    arr={m:[] for m in ['dAUROC','dAUPRC','dBrier']}
    # reset rng per model for reproducibility paired samples
    rr=np.random.default_rng(SEED)
    for b in range(B):
        idx=rr.integers(0,len(y),len(y)); yy=y[idx]
        if len(np.unique(yy))<2: continue
        arr['dAUROC'].append(roc_auc_score(yy,p1[idx])-roc_auc_score(yy,p0[idx]))
        arr['dAUPRC'].append(average_precision_score(yy,p1[idx])-average_precision_score(yy,p0[idx]))
        arr['dBrier'].append(brier_score_loss(yy,p1[idx])-brier_score_loss(yy,p0[idx]))
    deltas[name]={
      'dAUROC':summary[f'M1_{name}']['AUROC']-summary[f'M0_{name}']['AUROC'],
      'dAUPRC':summary[f'M1_{name}']['AUPRC']-summary[f'M0_{name}']['AUPRC'],
      'dBrier':summary[f'M1_{name}']['Brier']-summary[f'M0_{name}']['Brier'],
      'CI':{m:[float(np.quantile(a,.025)),float(np.quantile(a,.975))] for m,a in arr.items()}}

out={'n':len(y),'class_counts':{'0':int((y==0).sum()),'1':int((y==1).sum())},'summary':summary,'ci':boot_ci,'deltas':deltas,'details':results,
     'versions':{}}
import sys, sklearn, xgboost
out['versions']={'python':sys.version.split()[0],'sklearn':sklearn.__version__,'xgboost':xgboost.__version__}
with open('/mnt/data/nested_cv_results.json','w') as f: json.dump(out,f,indent=2)
print('\nSUMMARY')
for k,m in summary.items(): print(k, {a:round(m[a],4) for a in ['AUROC','AUPRC','Sensitivity','Specificity','Precision','F1','BalancedAccuracy','Brier','CalibrationIntercept','CalibrationSlope']})
print('\nDELTAS')
for k,v in deltas.items(): print(k,v)
print('TOTAL SEC',time.time()-start)
