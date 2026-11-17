import numpy as np
from scipy.special import logsumexp

def softmax(Z):
    return np.exp(Z-logsumexp(Z,axis=1,keepdims=True))

def entrenar(X,Y,epochs=1000,lr=0.01,beta=0.9,l2=0.001,ponderar=True):
    mu=X.mean(axis=0); sd=X.std(axis=0); sd=np.where(sd<1e-12,1,sd)
    A=np.column_stack([(X-mu)/sd,np.ones(len(X))])
    counts=Y.sum(axis=0)
    if np.any(counts==0): raise ValueError("Falta una clase en Train")
    pesos=len(Y)/(2*counts) if ponderar else np.ones(2)
    w=Y@pesos; W=np.zeros((A.shape[1],2)); acc=np.zeros_like(W); costos=[]
    for _ in range(epochs):
        Z=A@W; logp=Z-logsumexp(Z,axis=1,keepdims=True); P=np.exp(logp)
        costos.append(-np.sum(w[:,None]*Y*logp)/w.sum()+l2*np.sum(W[:-1]**2)/2)
        g=A.T@(w[:,None]*(P-Y))/w.sum(); g[:-1]+=l2*W[:-1]
        acc=beta*acc+(1-beta)*g**2
        W-=lr*g/(np.sqrt(acc)+1e-8)
    return dict(W=W,media=mu,desviacion=sd,pesos_clase=pesos),np.asarray(costos)

def predecir(X,modelo):
    A=np.column_stack([(X-modelo['media'])/modelo['desviacion'],np.ones(len(X))])
    P=softmax(A@modelo['W'])
    return P,P.argmax(axis=1)


# ----------------------------------------------------------------------------
# Ejecución: python train.py  (Softmax ponderado + L2 con RMSprop, SOLO Train)
# ----------------------------------------------------------------------------
def main():
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import dimr
    from config import GRAFICOS, SALIDA, cargar_mde

    X, Y, metadata, F, tr, te = cargar_mde()
    modelo_dim = dict(np.load(SALIDA / 'pesos.npz'))   # parámetros aprendidos por dimr.py
    Rtrain = dimr.transformar(X[tr], modelo_dim)

    hiper = dict(epochs=1000, lr=0.01, beta=0.9, l2=0.001, ponderar=True)
    modelo_soft, costos_soft = entrenar(Rtrain, Y[tr], **hiper)
    np.savez(SALIDA / 'pesoftmax.npz', **modelo_soft)
    hiper.update(dim_entrada=int(Rtrain.shape[1]), n_clases=2, n_train=int(tr.sum()))
    (SALIDA / 'hiperparametros_softmax.json').write_text(json.dumps(hiper, indent=2), encoding='utf-8')

    print('Pesos de clase:', modelo_soft['pesos_clase'])
    plt.figure(); plt.plot(costos_soft)
    plt.xlabel('Época'); plt.ylabel('Entropía cruzada ponderada + L2')
    plt.title('Convergencia Softmax — Train'); plt.grid(alpha=0.3)
    plt.savefig(GRAFICOS / 'convergencia_softmax.png', dpi=150, bbox_inches='tight'); plt.close()


if __name__ == '__main__':
    main()
