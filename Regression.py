import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_squared_error
import scipy
import math
import statsmodels.api as sm

# events
d1_load_miss='L1-dcache-load-misses'
d1_loads='L1-dcache-loads'
d1_prefetch_miss='L1-dcache-prefetch-misses'
d1_prefetch_loads='L1-dcache-prefetches'
d1_store_miss='L1-dcache-store-misses'
d1_stores='L1-dcache-stores'
i1_load_miss='L1-icache-load-misses'
i1_loads='L1-icache-loads'
llc_load_miss='LLC-load-misses'
llc_loads='LLC-loads'
branch_load_miss='branch-load-misses'
branch_loads='branch-loads'
branch_misses='branch-misses'
branches='branches'
cache_misses='cache-misses'
cache_refs='cache-references'
cycles='cycles'
dtlb_load_miss='dTLB-load-misses'
dtlb_loads='dTLB-loads'
dtld_store_miss='dTLB-store-misses'
dtlb_stores='dTLB-stores'
itlb_load_miss='iTLB-load-misses'
itld_loads='iTLB-loads'
instructions='instructions'
mem_loads='mem-loads'
backend_stalls='stalled-cycles-backend'
frontend_stalls='stalled-cycles-frontend'

class Result:
    "Store and output CPI stack coefficients and stats for a given benchmark"
    def __init__(self, name, events):
        id = name[:3]
        self.name = [n for n in names if id in n][0]
        self.events = events

    def SetRegressionResults(self, coeffs, rmse, r2, adj_r2, res):
        self.coeffs = coeffs
        self.rmse = rmse
        self.r2 = r2
        self.adj_r2 = adj_r2
        self.res = res
        
    def SetStats(self, events_sum, cyc_sum, ins_sum):
        self.events_sum = events_sum
        self.cyc_sum = cyc_sum 
        self.ins_sum = ins_sum

    def SetDataSize(self, numSamples):
        self.numSamples = numSamples

    def Dump(self, f):
        roundingFactor = 3
        WriteOp(f, self.name)
        WriteOp(f, self.numSamples)

        for c in self.coeffs:
            WriteOp(f, round(c, roundingFactor))

        WriteOp(f, self.rmse)
        WriteOp(f, self.r2)
        WriteOp(f, self.adj_r2)
        WriteOp(f, self.res)

        cpistacksum = self.coeffs[0]
        WriteOp(f, round(self.coeffs[0], roundingFactor))
        for i in range(len(self.events_sum)):
            cpishare = self.events_sum[i] * self.coeffs[i+1] / self.ins_sum
            cpistacksum += cpishare
            WriteOp(f, round(cpishare, roundingFactor))
        
        WriteOp(f, round(cpistacksum, roundingFactor))
        WriteOp(f, round(self.cyc_sum/self.ins_sum, roundingFactor), newline=True)        

    def WriteHeaders(self, f):

        WriteOp(f, 'Benchmark')
        WriteOp(f, 'Samples')

        WriteOp(f, 'Intercept')
        for i in self.events:
            WriteOp(f, i)

        WriteOp(f, 'RMSE')
        WriteOp(f, 'R2 Score')
        WriteOp(f, 'Adjusted R2 Score')
        WriteOp(f, 'Residual')

        WriteOp(f, 'Base CPI')
        for i in self.events:
            WriteOp(f, i + '_Share')           
        
        WriteOp(f, 'Sum Shares')
        WriteOp(f, 'Data Avg CPI', newline=True)

def WriteOp(op, data, newline=False):
    if newline:
        print(data, file=op)
    else:
        print(data, end=',',file=op)

# -- lambda functions for data transformations -- 

def get_cpi(row):
    return row[cycles] /row[instructions]

def get_ipc(row):
    return row[instructions]/row[cycles]

def get_norm(row, var):
    return row[var]/row[instructions]

def ComputeCPIIPC(rows):
    rows['cpi'] = rows.apply(get_cpi, axis=1)
    rows['ipc'] = rows.apply(get_ipc, axis=1)
    return rows

def NormalizeData(rows, eventvars, normvars):
    assert(len(normvars) == len(eventvars))
    # normalize the values per 'norm' instructions.    
    for i in range(len(eventvars)):
        rows[normvars[i]] = rows.apply(lambda row: get_norm(row, eventvars[i]), axis=1)
    return rows 

eventvars = [d1_load_miss, i1_load_miss, llc_load_miss, branch_misses, dtlb_load_miss, dtld_store_miss, itlb_load_miss]
def GetNormVars(vars):
    norm_vars = []
    for v in vars:
        n = v+'_norm'
        norm_vars.append(n)
    return norm_vars
    
def GetData(filename):
    #import the dataset
    curfile = os.path.join(os.path.curdir, filename)
    rows = pd.read_csv(curfile, sep='\s+')

    rows.drop(['CPU', 'THREAD', 'ENA', 'RUN'], axis=1, inplace=True)
    # print(rows.head())

    rows = rows.pivot(index='TIME', columns='EVENT', values='VAL')
    print("Number of data points: ", rows.shape)
    
    rows = rows.diff() # the event counts generated are cumulative
    rows = rows[1:] # drop first row
    return rows

def PlotData(rows):
    plotvars = GetNormVars([branch_misses, d1_load_miss, i1_load_miss, dtlb_load_miss, itlb_load_miss])
    plotvars.append('cpi')
    rows.plot(y=plotvars)
    plt.show()

def PrepareData(filename):
    rows = GetData(filename)
    rows = ComputeCPIIPC(rows)
    rows = NormalizeData(rows, eventvars, GetNormVars(eventvars))
    return rows

# -- Regression functions --

def GetAdjustedR2(R2, N, k):
    return (1 -(((1-R2) * (N - 1))/(N - k - 1)))

def GetL2Norm(x):
    dist = np.dot(x, x)
    return math.sqrt(dist)

def GetResidual(Ax, b):
    return GetL2Norm(Ax-b)

def DoLinearRegression(X_train, X_test, y_train, y_test):
    linreg = LinearRegression(normalize=False, fit_intercept=True, positive=True)
    linreg.fit(X_train, y_train)
    y_pred_nnls = linreg.predict(X_test)
    
    mse = mean_squared_error(y_test,y_pred_nnls)
    rmse = math.sqrt(mse/y_test.shape[0])
    r2_score_nnls = r2_score(y_test, y_pred_nnls)    

    # Print F-statistic and P Values
    X2 = sm.add_constant(X_train)
    y = y_train
    est = sm.OLS(y, X2)
    est2 = est.fit(positive=True)
    print(est2.summary())

    coeffs = [linreg.intercept_]
    for c in linreg.coef_:
        coeffs.append(c)
    return coeffs, rmse, r2_score_nnls, GetAdjustedR2(r2_score_nnls, X_test.shape[0],X_test.shape[1]), GetResidual(y_pred_nnls, y_test)

# -- top level methods --

def ComputeCPIStack(b, rows, results):
    X = rows[GetNormVars(eventvars)]
    y = rows['cpi']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    
    coeffs, rmse, r2, adj_r2, f1 = DoLinearRegression(X_train, X_test, y_train, y_test)
    results[b].SetRegressionResults(coeffs, rmse, r2, adj_r2, f1)

def CollectStatsForBenchmark(b, rows, results):    
    ins_sum = rows[instructions].sum()
    cyc_sum = rows[cycles].sum()
    
    events_sum = []
    for e in eventvars:
        var_sum = rows[e].sum()
        events_sum.append(var_sum)
    results[b].SetStats(events_sum, cyc_sum, ins_sum)
    results[b].SetDataSize(rows.shape[0])

def DumpResults(filename, results):
    with open('results.csv', 'w') as f:
        # write headers
        headerWritten = False
        for key in results:
            if not headerWritten:
                results[key].WriteHeaders(f)
                headerWritten = True
            Result.Dump(results[key], f)

if __name__ == "__main__":
    datafolder = 'data\\9events\\'
    benchmarks = ['523ref_200ms_samples.txt', '531ref_200ms_samples.txt','541ref_200ms_samples.txt','508ref_500ms_samples.txt','511ref_200ms_samples.txt','544ref_500ms_samples.txt']
    names = ['523.xalancbmk_r', '531.deepsjeng_r', '541.leela_r','508.namd_r','511.povray_r','544.nab_r']
    
    results = dict()
    for b in benchmarks:
        results[b] = Result(b, eventvars)
        # WriteOp(b)
        rows = PrepareData(datafolder + b)
        ComputeCPIStack(b, rows, results)
        CollectStatsForBenchmark(b, rows, results)
    
    DumpResults('results.csv', results)