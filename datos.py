import pandas as pd
import numpy as np

def datos(modo='entrena'):
  gid = 'd932a3cf4d6bdeef36a7230fff959301'
  tail = '64b604aedff376b7757b533d1c93685ce19b2077/bcdata'
  url = 'https://gist.githubusercontent.com/rodrgo/%s/raw/%s' % (gid, tail)
  df = pd.read_csv(url, sep=',')
  df = df.drop(columns=['Unnamed: 32', 'id'])
  var = 'diagnosis'
  df.loc[df[var] == 'M', [var]] = 1
  df.loc[df[var] == 'B', [var]] = 0
  X_cols = [c for c in df.columns if not c is var]
  X, y = df[X_cols].to_numpy(), df[var].to_numpy()
  idx = np.random.permutation(X.shape[0])
  train_idx, test_idx = idx[:69], idx[69:]
  idx = train_idx if modo == 'entrena' else test_idx
  return X[idx,:], y[idx]

