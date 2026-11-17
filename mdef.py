"""MDE para Auricle: ejecutar con python mdef.py (usa la carpeta del script).
Conserva la normalización NCDF por escala del código Bearing Diagnostic.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr

from config import FS, W, SENALES, PARTICIPANTES, EDA_DIR as EDA, RUTA, SALIDA, SEED


def multiescala(senal, tau_max=50):
    x = np.asarray(senal, dtype=float).ravel()
    if not 1 <= tau_max <= len(x) or not np.isfinite(x).all():
        raise ValueError('Escalas o señal inválidas')
    return {t: x[:len(x)//t*t].reshape(-1, t).mean(axis=1)
            for t in range(1, tau_max+1)}


def dispersion_entropy(x, d=3, tau_delay=1, c=3, referencia=None):
    """DE normalizada; d es dimensión de incrustación, no retardo."""
    x = np.asarray(x, dtype=float).ravel()
    if d < 1 or tau_delay < 1 or c < 2 or c**d > 1000000:
        raise ValueError('Parámetros de dispersión inválidos')
    M = len(x) - (d-1)*tau_delay
    if M <= 0 or not np.isfinite(x).all():
        raise ValueError('Serie demasiado corta o no finita')
    mu, sigma = (x.mean(), x.std()) if referencia is None else referencia
    if sigma == 0:
        return 0.0
    z = np.clip(np.floor(c*ndtr((x-mu)/sigma)).astype(int), 0, c-1)
    # Codificación base c, equivalente al bucle del archivo original.
    patrones = np.zeros(M, dtype=np.int64)
    for j in range(d):
        patrones += z[j*tau_delay:j*tau_delay+M] * c**j
    counts = np.bincount(patrones)
    p = counts[counts > 0] / M
    return float(-np.sum(p*np.log2(p))/np.log2(c**d))


def extraer_features(ventana, tau_max=50, d=3, tau_delay=1, c=3,
                     ncdf='por_escala'):
    ventana = np.asarray(ventana, dtype=float)
    if ventana.shape != (W, 2) or not np.isfinite(ventana).all() or np.any(ventana.std(axis=0)==0):
        raise ValueError('Ventana inválida')
    if ncdf not in ('por_escala', 'escala_1'):
        raise ValueError('NCDF desconocida')
    f = []
    for x in ventana.T:
        ref = (x.mean(), x.std()) if ncdf == 'escala_1' else None
        for serie in multiescala(x, tau_max).values():
            f.append(dispersion_entropy(serie, d, tau_delay, c, ref))
    return np.asarray(f)


def cargar_catalogo(ruta):
    carpeta = Path(ruta)/EDA
    archivo = carpeta/'ventanas_finales.csv'
    if not archivo.exists():
        archivo = carpeta/'detalle_ventanas.csv'
        if not archivo.exists():
            raise FileNotFoundError('Ejecuta la celda de exportación de ventanas_finales.csv del EDA')
        df = pd.read_csv(archivo)
        df = df.loc[df.Valida_basica & ~df.Posible_saturacion].copy()
    else:
        df = pd.read_csv(archivo)
    df = df.sort_values(['Participante','Inicio_muestra']).reset_index(drop=True)
    if 'Ventana_ID' not in df:
        df.insert(0,'Ventana_ID',np.arange(len(df)))
    if set(df.Participante) != set(PARTICIPANTES):
        raise ValueError('Participantes distintos de los acordados')
    if df.Ventana_ID.duplicated().any() or df.duplicated(['Participante','Inicio_muestra']).any():
        raise ValueError('Ventanas duplicadas')
    if not (df.Fin_muestra-df.Inicio_muestra == W).all() or not (df.Inicio_muestra >= 0).all():
        raise ValueError('Índices inválidos')
    if not df.Etiqueta.isin([0,1]).all() or not (df.Actividad.map({'rest':0,'valsalva':1}) == df.Etiqueta).all():
        raise ValueError('Etiquetas inconsistentes')
    if not (df.groupby('Participante').Etiqueta.nunique()==2).all():
        raise ValueError('Falta una clase en algún participante')
    if 'Valida_basica' in df and not df.Valida_basica.all():
        raise ValueError('Catálogo contiene ventanas inválidas')
    if 'Posible_saturacion' in df and df.Posible_saturacion.any():
        raise ValueError('Catálogo contiene ventanas con alertas')
    return df


def dividir(df, seed=42):
    sujetos = np.random.default_rng(seed).permutation(PARTICIPANTES)
    train, test = sujetos[:9].tolist(), sujetos[9:].tolist()
    df = df.copy()
    df['Split'] = np.where(df.Participante.isin(train),'train','test')
    return df, train, test


def extraer_catalogo(df, ruta, **params):
    import wfdb
    F = np.empty((len(df), 2*params['tau_max']))
    for sujeto, grupo in df.groupby('Participante', sort=False):
        print('Procesando:', sujeto, flush=True)
        base = str(Path(ruta)/sujeto)
        canales = SENALES + ['Activity_Annotations']
        rec = wfdb.rdrecord(base, channel_names=canales, physical=True)
        if rec.fs != FS or list(rec.sig_name) != canales:
            raise ValueError(f'{sujeto}: frecuencia o canales incorrectos')
        for i, fila in grupo.iterrows():
            a,b = int(fila.Inicio_muestra),int(fila.Fin_muestra)
            v = rec.p_signal[a:b]
            codigo = 0 if fila.Etiqueta == 0 else 2
            if v.shape != (W,3) or not np.allclose(v[:,2],codigo,atol=1e-5,rtol=0):
                raise ValueError(f'{sujeto}: anotación o longitud incorrecta en {a}')
            F[i] = extraer_features(v[:,:2], **params)
        del rec
    return F


def ejecutar(ruta, salida=None, tau_max=50, d=3, tau_delay=1, c=3,
             seed=42, ncdf='por_escala'):
    ruta = Path(ruta)
    salida = Path(salida) if salida else ruta/'resultados_mde'
    df, train, test = dividir(cargar_catalogo(ruta), seed)
    params = dict(tau_max=tau_max,d=d,tau_delay=tau_delay,c=c,ncdf=ncdf)
    print('Train:',train,'\nTest:',test)
    F = extraer_catalogo(df, ruta, **params)
    tr = df.Split.eq('train').to_numpy()
    y = df.Etiqueta.to_numpy(dtype=int)
    Y = np.eye(2,dtype=int)[y]
    mu, sigma = F[tr].mean(axis=0), F[tr].std(axis=0)
    sigma_segura = np.where(sigma==0,1,sigma)
    Z = (F-mu)/sigma_segura
    nombres = [f'{s}_MDE_{t:02d}' for s in SENALES for t in range(1,tau_max+1)]
    salida.mkdir(parents=True,exist_ok=True)
    def guardar(nombre, matriz):
        pd.DataFrame(matriz).to_csv(salida/nombre,index=False,header=False)
    guardar('dfeatures.csv', Z)
    guardar('dclass.csv', Y)
    guardar('dtrain.csv', np.column_stack([Z[tr],Y[tr]]))
    guardar('dtest.csv', np.column_stack([Z[~tr],Y[~tr]]))
    pd.DataFrame(F,columns=nombres).to_csv(salida/'features_originales.csv',index=False)
    df.to_csv(salida/'metadata.csv',index=False)
    df.loc[tr].to_csv(salida/'metadata_train.csv',index=False)
    df.loc[~tr].to_csv(salida/'metadata_test.csv',index=False)
    np.savez(salida/'escalador.npz',media=mu,desviacion=sigma_segura,desviacion_original=sigma)
    (salida/'config.json').write_text(json.dumps(dict(**params,seed=seed,fs=FS,W=W,train=train,test=test,features=nombres,clases=['rest','valsalva']),indent=2),encoding='utf-8')
    print(pd.crosstab(df.Split,df.Actividad))
    print('dfeatures:',Z.shape,'dclass:',Y.shape)
    print('dtrain:',(tr.sum(),Z.shape[1]+2),'dtest:',((~tr).sum(),Z.shape[1]+2))
    print('Resultados:',salida)
    return Z,Y,df


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ruta',default=str(RUTA))
    p.add_argument('--salida',default=str(SALIDA/'mde'))
    p.add_argument('--tau-max',type=int,default=50)
    p.add_argument('--d',type=int,default=3)
    p.add_argument('--tau-delay',type=int,default=1)
    p.add_argument('--c',type=int,default=3)
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--ncdf',choices=['por_escala','escala_1'],default='por_escala')
    ejecutar(**vars(p.parse_args()))
