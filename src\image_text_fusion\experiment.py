import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def dataset(n=2200,seed=27):
 r=np.random.default_rng(seed);latent=r.normal(size=(n,5));image=latent@r.normal(size=(5,10))+r.normal(size=(n,10));text=latent@r.normal(size=(5,8))+r.normal(size=(n,8));logit=.8*latent[:,0]-.7*latent[:,1]+.6*latent[:,3];y=r.binomial(1,1/(1+np.exp(-logit)));return image,text,y
def evaluate(seed=27):
 i,t,y=dataset(seed=seed);cut=1500;sets={"image_only":i,"text_only":t,"fused":np.c_[i,t]};out={}
 for name,x in sets.items():
  m=LogisticRegression(max_iter=500).fit(x[:cut],y[:cut]);p=m.predict_proba(x[cut:])[:,1];out[name]={"auroc":float(roc_auc_score(y[cut:],p))}
 m=LogisticRegression(max_iter=500).fit(sets["fused"][:cut],y[:cut]);zero_i=np.c_[np.zeros_like(i[cut:]),t[cut:]];zero_t=np.c_[i[cut:],np.zeros_like(t[cut:])];out["ablation"]={"without_image_auroc":float(roc_auc_score(y[cut:],m.predict_proba(zero_i)[:,1])),"without_text_auroc":float(roc_auc_score(y[cut:],m.predict_proba(zero_t)[:,1]))};return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
