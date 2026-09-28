from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support

SEED=42
BATCH_SIZE=64
LR=0.001
EPOCHS=8
VAL_SIZE=5000
DATA=Path("data")
OUT=Path("results")
OUT.mkdir(exist_ok=True)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

class MNISTNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net=nn.Sequential(nn.Flatten(), nn.Linear(784,hidden), nn.ReLU(), nn.Linear(hidden,10))
    def forward(self,x): return self.net(x)

def loaders():
    tf=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    tr=datasets.MNIST(DATA,train=True,download=True,transform=tf)
    te=datasets.MNIST(DATA,train=False,download=True,transform=tf)
    train,val=random_split(tr,[len(tr)-VAL_SIZE,VAL_SIZE],generator=torch.Generator().manual_seed(SEED))
    kw=dict(batch_size=BATCH_SIZE,num_workers=0)
    return DataLoader(train,shuffle=True,**kw),DataLoader(val,shuffle=False,**kw),DataLoader(te,shuffle=False,**kw)

def epoch(model,loader,loss_fn,opt=None):
    train=opt is not None
    model.train() if train else model.eval()
    loss_sum=correct=n=0
    with torch.set_grad_enabled(train):
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            if train: opt.zero_grad()
            z=model(x); loss=loss_fn(z,y)
            if train: loss.backward(); opt.step()
            loss_sum+=loss.item()*len(y); correct+=(z.argmax(1)==y).sum().item(); n+=len(y)
    return loss_sum/n,correct/n

def train(hidden,tl,vl):
    m=MNISTNet(hidden).to(DEVICE); loss_fn=nn.CrossEntropyLoss(); opt=torch.optim.Adam(m.parameters(),lr=LR)
    h={"train_loss":[],"val_loss":[],"train_accuracy":[],"val_accuracy":[]}
    for e in range(EPOCHS):
        a,b=epoch(m,tl,loss_fn,opt); c,d=epoch(m,vl,loss_fn)
        h["train_loss"].append(a); h["val_loss"].append(c); h["train_accuracy"].append(b); h["val_accuracy"].append(d)
        print(f"Epoch {e+1}/{EPOCHS} | train_acc={b:.4f} | val_acc={d:.4f}")
    return m,h

def evaluate(m,loader):
    m.eval(); ys=[]; ps=[]; total=0; n=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE); z=m(x)
            total+=nn.functional.cross_entropy(z,y,reduction="sum").item(); n+=len(y)
            ys.extend(y.cpu().numpy()); ps.extend(z.argmax(1).cpu().numpy())
    p,r,f,_=precision_recall_fscore_support(ys,ps,average="weighted",zero_division=0)
    return {"test_loss":total/n,"test_accuracy":accuracy_score(ys,ps),"precision_weighted":p,"recall_weighted":r,"f1_weighted":f,"cm":confusion_matrix(ys,ps).tolist(),"y":ys,"p":ps}

def plots(name,h,ev):
    x=range(1,EPOCHS+1)
    for metric,title in [("loss","Loss"),("accuracy","Accuracy")]:
        plt.figure(figsize=(8,5)); plt.plot(x,h["train_"+metric],label="Training"); plt.plot(x,h["val_"+metric],label="Validation")
        plt.xlabel("Epoch"); plt.ylabel(title); plt.title(f"{name} — {title}"); plt.legend(); plt.tight_layout()
        plt.savefig(OUT/f"{name.lower()}_training_{metric}.png",dpi=160); plt.close()
    plt.figure(figsize=(8,6)); sns.heatmap(np.array(ev["cm"]),annot=True,fmt="d",cmap="Blues",cbar=False)
    plt.xlabel("Predicted Label"); plt.ylabel("True Label"); plt.title(f"{name} — Confusion Matrix"); plt.tight_layout()
    plt.savefig(OUT/f"{name.lower()}_confusion_matrix.png",dpi=160); plt.close()
    report=classification_report(ev["y"],ev["p"],digits=4,zero_division=0)
    (OUT/f"{name.lower()}_classification_report.txt").write_text(report)

def main():
    seed(); tl,vl,te=loaders()
    results={}; rows=[]
    for name,hidden in [("Baseline",128),("Modified",256)]:
        seed(); m,h=train(hidden,tl,vl); ev=evaluate(m,te); plots(name,h,ev)
        results[name]={"hidden_neurons":hidden,"history":h,"evaluation":{k:v for k,v in ev.items() if k not in ("y","p")}}
        rows.append({"model":name,"hidden_neurons":hidden,**results[name]["evaluation"]})
    df=pd.DataFrame(rows); df.to_csv(OUT/"comparison.csv",index=False)
    ax=df.plot(x="model",y="test_accuracy",kind="bar",legend=False,figsize=(8,5)); ax.set_ylabel("Test Accuracy"); ax.set_ylim(0,1); plt.tight_layout(); plt.savefig(OUT/"model_comparison.png",dpi=160); plt.close()
    (OUT/"metrics.json").write_text(json.dumps({"device":str(DEVICE),"batch_size":BATCH_SIZE,"learning_rate":LR,"epochs":EPOCHS,"results":results},indent=2))
    print("\nFINAL COMPARISON\n",df.to_string(index=False))
    print("\nResults saved to",OUT.resolve())

if __name__=="__main__":
    main()
