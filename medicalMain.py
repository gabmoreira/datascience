'''
    File name: Exam.ipynb
    Author: Gabriel Moreira
    Date last modified: 14/10/2018
    Python Version: 3.6.5
'''
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import t
from matplotlib import gridspec
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.linear_model import LogisticRegression

# Data extration

plt.rcParams['figure.dpi'] = 130
dataframe=pandas.read_csv(r"/data/MedicalData1.csv",sep=';',decimal=',')

listColNames=list(dataframe.columns)

XY=dataframe.values
ColNb_Y=listColNames.index('Disease progression')

Y=XY[:,ColNb_Y].reshape((XY.shape[0],1))   #reshape is to make sure that Y is a column vector
X = np.delete(XY, ColNb_Y, 1)

X_scaled = preprocessing.scale(X)

[n_lines, n_col] = X.shape

listColNames.pop(ColNb_Y)     #to make it contains the column names of X only

for Col in range(len(listColNames)):
    plt.plot(X[:,Col],Y[:],'.')
    plt.xlabel(listColNames[Col])
    plt.ylabel('Disease progression')
    plt.show()
    
"""
    Correlations between th 18 variables 
    and Disease Progression
"""
R2 = []
lr = LinearRegression()
for Col in range(X.shape[1]):
    x = X[:,Col].reshape(XY.shape[0], 1)
    lr.fit(x,Y)
    R2.append(lr.score(x,Y))
    
markerline, stemlines, baseline = plt.stem(np.array([i for i in range(n_col)]), np.array(R2), '-.')
plt.xticks(np.arange(len(listColNames)), listColNames, rotation='vertical');
plt.ylabel('Correlation (R2)')
plt.title('Correlation with Disease Progression')
plt.setp(baseline, 'color', 'navy', 'linewidth', 1)
plt.setp(stemlines, 'color', 'navy', 'linewidth', 1, 'alpha', 0.5);
plt.setp(markerline, 'color', 'navy', 'markersize', 3, 'linewidth', 1);

Acid1DenIndex = listColNames.index('Acid 1 density')
Acid1Den = X[:,Acid1DenIndex].reshape(XY.shape[0], 1)
lr.fit(Acid1Den, Y[:])

print('> R2 DiseaseProgression(Acid1Density) = ' + str(round(lr.score(Acid1Den, Y[:]),4)))

"""
    Evaluates the prediction stability
"""
kf = KFold(n_splits=4)
eSplit = np.zeros(4)
b0 = np.zeros(4)
b1 = np.zeros(4) 
i = 0

for train, test in kf.split(Acid1Den):
    lr.fit(Acid1Den[train], Y[train])
    eSplit[i] = np.mean(abs(np.subtract(Y, lr.predict(Acid1Den))))
    b1[i] = lr.coef_
    b0[i] = lr.intercept_
    i += 1
    
plt.figure()
plt.plot([i for i in range(1,5)], eSplit, color='navy')
plt.title('eSplit (Y~Acid 1 Density)')
plt.xlabel('i-th Fold')
plt.ylabel('eSplit')
plt.axhline(np.mean(eSplit), color='lightgray')
plt.text(2.7, np.mean(eSplit), 'mean eSplit', color='gray')

print('K-FOLDS (k=4) and linear regression for Disease Progression ~ Acid 1 Density\n')
print('> Mean eSplit = %.2f' % round(np.mean(eSplit),2))
print('> Std Variation (b1) = %.2f' % round(np.sqrt(np.var(b1)),2));


"""
    Uses K-folds with K=4 and a linear regression on 
    Biomarker 8 and Disease Progression.
    Prints the R2, Mean eSplit and the std_var(b1).
    Plots the eSplit for each fold.
"""

Biomarker8 = X[:,listColNames.index('Biomarker 8')].reshape([len(Y),1])
lr.fit(Biomarker8[:], Y[:])
print('> R2 Progression(Biomarker 8) = ', str(round(lr.score(Biomarker8[:], Y[:]),4)))
      
i = 0
for train, test in kf.split(Biomarker8):
    lr.fit(Biomarker8[train], Y[train])
    b0[i] = lr.intercept_
    b1[i] = lr.coef_
    eSplit[i] = np.mean(abs(np.subtract(Y, lr.predict(Biomarker8))))
    i += 1
    
fig = plt.figure(figsize=(9, 3)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.plot([i for i in range(1,5)], eSplit, color='navy')
ax0.set_title('eSplit (Y~Biomarker8)')
ax0.set_xlabel('i-th Fold')
ax0.set_ylabel('eSplit')
ax0.axhline(np.mean(eSplit), color='lightgray')
ax0.text(2.7, np.mean(eSplit), 'mean eSplit', color='gray')

print('> Mean eSplit = %.2f' % round(np.mean(eSplit),4))
print('> Std Variation (b1) = %.4f\n' % round(np.sqrt(np.var(b1)),4))

"""
    Uses K-folds with K=4 and a linear regression on 
    Pressure1 and Disease Progression.
    Prints the R2, Mean eSplit and the std_var(b1)
    Plots the eSplit for each fold
"""

Pressure1 = X[:,listColNames.index('Pressure 1')].reshape([len(Y),1])
lr.fit(Pressure1[:], Y[:])
print('> R2 Progression(Pressure 1) = ', str(round(lr.score(Pressure1[:], Y[:]),4)))

i = 0
for train, test in kf.split(Pressure1):
    lr.fit(Pressure1[train], Y[train])
    b0[i] = lr.intercept_
    b1[i] = lr.coef_
    eSplit[i] = np.mean(abs(np.subtract(Y, lr.predict(Pressure1))))
    i += 1


ax1.plot([i for i in range(1,5)], eSplit, color='navy')
ax1.set_title('eSplit (Y~Pressure1)')
ax1.set_label('i-th Fold')
ax1.set_ylabel('eSplit')
ax1.set_xlabel('i-th Fold')
ax1.axhline(np.mean(eSplit), color='lightgray')
ax1.text(2.7, np.mean(eSplit), 'mean eSplit', color='gray')

print('> Mean eSplit = %.2f' % round(np.mean(eSplit),4))
print('> Std Variation (b1) = %.2f' % round(np.sqrt(np.var(b1)),4))

""" 
    This section has the functions used in the statistic t-test.
"""

def ss(x, y):
    """ 
        Calculates the residual variance.
    """
    n = len(x)
    lr.fit(x[:], y[:])
    x = x.reshape([n,1])
    y = y.reshape([n,1])
    s = np.sqrt(np.sum(np.power(np.subtract(y, lr.predict(x)), 2)) / (n-2))
    return s

def sx2(x):
    """ 
        Calculates the variance of the variable X, diving by (n-2) 
        instead of n.
    """
    n = len(x)
    x = x.reshape([n,1])
    sx2 = (1.0/(n-2))*np.sum(np.power(np.subtract(x, np.array([np.mean(x)]*n).reshape([n,1])),2))
    return sx2

def testStat(x,y, alpha):
    """ 
        Computes a statistical test on the hypothesis that Beta-1 is zero.
        Type-I error being equal to alpha.
    """
    n = len(x)
    x = x.reshape([n,1])
    y = y.reshape([n,1])
    lr.fit(x[:], y[:])
    b1 = lr.coef_
    interval = ss(x, y)*np.sqrt((1.0/((n-1)*sx2(x))))*t.ppf(1-alpha/2.0, n-2)
    pvalue = t.sf(abs(b1 / (ss(x, y)*np.sqrt(1/((n-1)*sx2(x))))), n-2)
    print('> H0: Beta1 = 0. Type-I error (alpha) = %.2f.' % alpha)
    print('> p-value = %.4E.' % pvalue)
    print('> b1 = %.5f.' % b1[0])
    print('> Confidence interval (%.f percent) = [%.5f, +%.5f].' % (100-alpha*100, 
          -interval, interval))
    if pvalue < alpha:
        print('> H0 rejected. We can assume Beta-1 is different than 0.')
    else:
        print('> We cannot reject H0.')

""" 
    t-tests with confidence intervals and p-values to determine 
    whether or not we can reject the null hypothesis (H0).
    H0: The slope (Beta1) is null - there is no significative relationship.
"""

print('Biomarker 8:')
testStat(Biomarker8, Y, 0.05)
print('\nPressure 1:')
testStat(Pressure1, Y, 0.05)
print('\nAcid 1 Density:')
testStat(Acid1Den, Y, 0.05)

def hii(x):
    """ 
        Calculates the for the diagonal H matrix.
        Ŷ = HY
    """
    n = len(x)
    x = x.reshape([n,1])
    x_mean = np.array([np.mean(x)]*n)
    num = np.matmul(np.subtract(x.reshape([n,1]), x_mean.reshape(n,1)), 
                    np.subtract(x.reshape([1,n]), x_mean.reshape(1,n)))
    den = np.sum(np.square(np.subtract(x, x_mean)))
    h = np.array([1/n]*n**2).reshape(n,n) + np.divide(num,den)
    h_diag = np.diagonal(h)
    return h_diag

Biomarker5 = np.array(X[:,listColNames.index('Biomarker 5')])
n = len(Biomarker5)
Biomarker5 = Biomarker5.reshape([n ,1])

h_diag = hii(Biomarker5)
plt.figure()
plt.axhline(np.mean(h_diag), color='lightgray', linewidth=0.5)
plt.text(-10, np.mean(h_diag), ' average diag(h)', color='gray')
plt.scatter(Biomarker5, h_diag, alpha=0.6, color='navy')
plt.title('Outlier Identification using H-Matrix')
plt.xlabel('Biomarker 5')
plt.ylabel('diag(H) - Outlier Detection');

""" 
    Functions used in calculation of the 
    Cooks Distance and to find the outliers.
"""

def residue(x,y):
    """ 
        Calculates the residues vector resultant from 
        a linear model being fitted to the pairs (x,y).
    """
    n = len(x)
    x = x.reshape(n, 1)
    y = y.reshape(n, 1)
    lr.fit(x,y)
    res = np.subtract(y, lr.predict(x))
    
    return res


def res_i(x,y):
    """ 
        Calculates the standardised residues vector resultant 
        from a linear model being fitted to the pairs (x,y).
    """
    n = len(x)
    x = x.reshape([n,1])
    y = y.reshape([n,1])
    num = residue(x,y)
    den = np.multiply(ss(x,y), np.sqrt(np.subtract(np.ones(n), hii(x))))
    res_std = np.divide(num.reshape([n,1]), den.reshape([n,1]))
    
    return res_std


def cook(x, y):
    """ 
        Calculates the cook's distances.
    """
    n = len(x)
    x = x.reshape([n,1])
    y = y.reshape([n,1])
    num = hii(x)
    den = np.multiply(np.ones(n)*2, np.subtract(np.ones(n), hii(x)))
    cook = np.multiply(np.divide(num,den).reshape([n,1]), np.square(res_i(x,y)))
    
    return cook


def outlierFinder(x, y):
    """ 
        Determines the outliers in a dataset, using the cook's distance
        and a cutoff criterion of 4/n.
        Nominal_cook is the array containing the data not classified as
        outliers.
        <...>_ind are the indexes for the outliers and the nominal data. 
    """  
    n = len(x)
    cookd = cook(x,y)
    nominal_cook = cookd[np.where(cookd <= 4/n)]
    outliers_cook = cookd[np.where(cookd > 4/n)]
    nominal_ind = np.where(cookd <= 4/n)
    outlier_ind = np.where(cookd > 4/n) 
    
    return [nominal_cook, outliers_cook, nominal_ind, outlier_ind]

y = cook(Biomarker5, Y)

plt.figure()
plt.axhline(y=0, color='gray', linewidth=0.2)
plt.scatter(Biomarker5, cook(Biomarker5, Y), color='navy', alpha=0.7)
plt.title('Cooks Distance for each observation of Biomarker 5')
plt.xlabel('Biomarker 5')
plt.ylabel('Cook''s Distance');

""" 
    Represents, graphically, the outliers and the rest of the
    data in different colors and for their Cooks distance.
    Uses a cutoff of 4/n.
"""

[nominal_cook, outliers_cook, nominal_ind, outliers_ind] = outlierFinder(Biomarker5, Y)
plt.figure()
plt.plot(outliers_cook,np.zeros_like(outliers_cook), 'o', color='orange', alpha=0.8)
plt.plot(nominal_cook,np.zeros_like(nominal_cook), 'o', color='navy', alpha=0.3)
plt.title('Outlier identification using Cooks distance')
plt.xlabel('Cook''s distance')
frame1 = plt.gca()
frame1.axes.get_yaxis().set_visible(False)
plt.legend(['Outliers (4/n cutoff)', 'Rest of the data points', ])
plt.axvline(x=4/n, linewidth=0.2, color='black', linestyle='--')
plt.text(4/n,0, '  cutoff', color='gray', rotation='vertical');

"""
    Linear regression between Disease Progression and Biomarker 5
    Uses Cook's distances to identify Outliers and color-mapping 
    to represent them differently.
    Also plots the standardised residues.
"""

# Linear Regression with the identified outliers present
lr.fit(Biomarker5, Y)
R2 = round(lr.score(Biomarker5, Y), 4)
text = 'R2 = ' + str(R2)

# Calculates the residues array
res = res_i(Biomarker5, Y)

# Divides the figure two represent the two plots
fig = plt.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

# Plots stuff - different colors for outliers and normal data
ax0.plot(Biomarker5[outliers_ind[0]], Y[outliers_ind[0]], 'v', color='orange')
ax0.plot(Biomarker5[nominal_ind[0]], Y[nominal_ind[0]], 'o', color='navy', alpha=0.5)
ax0.plot(Biomarker5, lr.predict(Biomarker5), color='navy', alpha=0.8)
ax0.set_title('Disease Progression vs. Biomarker 5 (complete)')
ax0.set_ylabel('Disease Progression')
ax0.legend(['Outliers (4/n cutoff)','Rest of the data points'], loc='upper left')
ax0.text(x=-2.5, y=16, s=text, color='gray')
ax0.xaxis.set_major_locator(plt.NullLocator())
ax0.xaxis.set_major_formatter(plt.NullFormatter())
markerline1, stemlines1, baseline1 = plt.stem(Biomarker5[outliers_ind[0]],res[outliers_ind[0]], '-.')
markerline2, stemlines2, baseline2 = plt.stem(Biomarker5[nominal_ind[0]],res[nominal_ind[0]], '-.')
ax1.set_xlabel('Biomarker 5')
ax1.set_ylabel('Residues std (r_i)')
plt.setp(baseline1, 'color', 'orange', 'linewidth', 0)
plt.setp(stemlines1, 'color', 'orange', 'linewidth', 1);
plt.setp(markerline1, 'color', 'orange', 'markersize', 3, 'linewidth', 1, 'alpha', 0.5);
plt.setp(baseline2, 'color', 'navy', 'linewidth', 0)
plt.setp(stemlines2, 'color', 'navy', 'linewidth', 1, 'alpha', 0.5);
plt.setp(markerline2, 'color', 'navy', 'markersize', 3, 'linewidth', 1);
plt.axhline(y=0, color='navy')

"""
    Linear regression between Disease Progression and Biomarker 5 without outliers.
    Uses Cook's distances to identify Outliers and color-mapping 
    to represent them differently.
    Also plots the standardised residues.
"""

# Remove the Outliers from the dataset
Biomarker5_clean = np.delete(Biomarker5, outliers_ind[0]).reshape([len(Biomarker5)-len(outliers_cook),1])
Y_clean = np.delete(Y, outliers_ind[0]).reshape([len(Y)-len(outliers_cook),1])

# Calculates the residues array
res = res_i(Biomarker5_clean, Y_clean)

# Linear regression performed on the clean dataset
lr.fit(Biomarker5_clean, Y_clean)
R2_clean = round(lr.score(Biomarker5_clean, Y_clean),4)
text = 'R2 = ' + str(R2_clean)

# Divides the figure two represent the two plots
fig = plt.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

# Plots stuff
ax0.set_title('Disease Progression vs. Biomarker 5 (without outliers)')
ax0.set_ylabel('Disease Progression')
ax0.text(x=-2, y=16, s=text, color='gray')
ax0.plot(Biomarker5_clean, lr.predict(Biomarker5_clean), color='navy', alpha=0.8)
ax0.plot(Biomarker5_clean, Y_clean, 'o', color='navy', alpha=0.5)
ax0.xaxis.set_major_locator(plt.NullLocator())
ax0.xaxis.set_major_formatter(plt.NullFormatter())
ax1.set_xlabel('Biomarker 5')
ax1.set_ylabel('Residues std (r_i)')
markerline, stemlines, baseline = plt.stem(Biomarker5_clean,res, '-.')
plt.setp(baseline, 'color', 'navy', 'linewidth', 0)
plt.setp(stemlines, 'color', 'navy', 'linewidth', 1, 'alpha', 0.5);
plt.setp(markerline, 'color', 'navy', 'markersize', 3, 'linewidth', 1);
plt.axhline(y=0, color='navy');

""" 
    Preliminary Study on the influence of the alpha parameter (LASSO)
    has on the correlation coefficient R2 and the number of 
    selected / unselected variables, 
    i.e., number of null coefficients in the Beta vector.
"""

alpha = [0.05+0.01*i for i in range(1, 101)]
lassoR2 = []
NumEliminatedVars = []

for i in range(100):
    lassoReg = Lasso(alpha = alpha[i])
    lassoReg.fit(X_scaled, Y)
    lassoR2.append(round(lassoReg.score(X_scaled, Y), 2))
    NumEliminatedVars.append(np.size(np.where(lassoReg.coef_ == 0)))

# Plots more stuff
fig, ax1 = plt.subplots()
ax1.set_title('R2 and No. of eliminated variables vs. alpha')
ax1.set_xlabel('alpha (LASSO)')
ax1.set_ylabel('No. of eliminated variables', color='green')
ax1.plot(alpha, NumEliminatedVars, '-', alpha=0.8, color='green')
ax1.tick_params(axis='y', labelcolor='green')
ax2 = ax1.twinx()  
ax2.set_ylabel('R2', color='navy')  
ax2.plot(alpha, lassoR2, '-', alpha=0.8, color='navy')
ax2.tick_params(axis='y', labelcolor='navy')
fig.tight_layout()  
plt.show()

def lassoVarSelect(X, Y, theta):
    """
        Selects the variables using Lasso Regression and for a correlation
        coefficient of theta*R2(Linear Regression).
        Varies the alpha automatically and stops iterating once it reaches the
        desired R2 (Without using cross-validation).
    """
    
    # First a mulitple linear regression is performed to find the R2 max
    lr = LinearRegression()
    lr.fit(X,Y) 
    R2Goal = theta*lr.score(X,Y)   
    alpha = [0.05+0.01*i for i in range(1, 101)]
    
    # Increasing alpha until the R2 goal is achieved
    for i in range(100):
        lassoReg = Lasso(alpha = alpha[i])
        lassoReg.fit(X_scaled, Y)
        lassoR2 = lassoReg.score(X_scaled, Y)
        if  lassoR2 <= R2Goal:
            break
            
    return [lassoR2, lassoReg.coef_]

# Selecting variables with a R2 cutoff of 95% of the value for linear multiple regression
theta = 0.85 # Adjust here! (85% of the R2 for alpha = 0)
[R2, betaVector] = lassoVarSelect(X_scaled,Y,theta)
numSelectVar = np.size(np.where(betaVector != 0))

print('> R2 = %.4f' % round(R2,4))
print('> Number of selected variables: %d' % numSelectVar)

def optimizeSelection(X,Y, alpha, method):
    """
        Method can be linear or log for logistic regression
        Uses Leave-one-out and Lasso regression while iterating over 
        multiple alphas.
        Saves the Betas Coefs using in each alpha-iteration and in 
        each LOO-iteration
        Outputs: eSplit, the number of variables selected and the beta 
        vector in each iteration
    """
    
    loo = LeaveOneOut()
    num_itera = len(alpha)
    eSplit = np.empty((num_itera, loo.get_n_splits(X)))
    NumVarSelect = np.empty((num_itera, loo.get_n_splits(X)))
    Beta = np.empty((num_itera, loo.get_n_splits(X), np.shape(X)[1]))
        
    if method == 'linear':
        
        for i in range(num_itera):
            lassoReg = Lasso(alpha = alpha[i])
            j = 0
            for train_index, test_index in loo.split(X):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                lassoReg.fit(X_train, Y_train)
                Beta[i][j][:] = lassoReg.coef_
                NumVarSelect[i][j] = np.size(Beta[i][j])-np.size(np.where(Beta[i][j] == 0))
                Y_hat = lassoReg.predict(X_test)
                eSplit[i][j] = abs(Y_hat - Y_test)
                j+=1
                
    elif method == 'log':
        
        for i in range(num_itera):
            logReg = LogisticRegression(penalty='l1', C=alpha[i])
            logReg.fit(X,Y.reshape(n_lines,))
            j = 0
            for train_index, test_index in loo.split(X):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                logReg.fit(X_train, Y_train)
                Beta[i][j][:] = logReg.coef_
                NumVarSelect[i][j] = np.size(Beta[i][j])-np.size(np.where(Beta[i][j] == 0))
                Y_hat = logReg.predict(X_test)
                eSplit[i][j] = abs(Y_hat - Y_test)
                j+=1
                   
    return [eSplit, NumVarSelect, Beta]

"""
    Calls the function optimizeSelection using the 
    specified alpha vector for the iteration.
"""

alpha = [0.05+0.01*i for i in range(1, 101)]
[eSplit, NumVarSelect, Beta] = optimizeSelection(X_scaled,Y, alpha, method='linear')

"""
    Calculates the average eSplit for each alpha and the 
    average number of selected variables for each alpha
"""

eSplitMean = np.mean(eSplit, axis=1)
NumVarSelectMean = np.round(np.mean(NumVarSelect, axis=1))

"""
    Finds the minimum eSplit that verifies number of selected variables <= 3,
    the corresponding alpha and the actual number of selected variables
"""
# find the minimum eSplit with max 3 vars selected
eSplit_min = np.min(eSplitMean[np.where(NumVarSelectMean<=3)])

# find the alpha that generates the lowest eSplit for max 3 vars selected
alpha_min = alpha[np.where(eSplitMean == eSplit_min)[0].tolist()[0]]

# find the number of variables satisfying: number of selected vars <= 3 and lowest eSplit
NumVarSelect_min = NumVarSelectMean[np.where(eSplitMean == eSplit_min)[0].tolist()[0]]

"""
    Shows the results of the optimization graphically
"""

fig, ax1 = plt.subplots()
plt.axvline(x=alpha_min, color='lightgray', linestyle='--')
ax1.axhline(y=eSplit_min, color='lightgray', linestyle='--')
ax1.set_title('eSplit and Num. selected variables vs. alpha (LOO)')
ax1.set_xlabel('alpha (LASSO)')
ax1.set_ylabel('Mean eSplit', color='green')
ax1.plot(alpha, eSplitMean, alpha=0.8, color='green')
ax1.plot(alpha_min, eSplit_min,'o', color='green')
ax1.tick_params(axis='y', labelcolor='green')
ax1.text(alpha_min, eSplit_min,'  eSplit min', color='green')
ax2 = ax1.twinx()
ax2.axhline(y=NumVarSelect_min, color='lightgray', linestyle='--')
ax2.set_ylabel('Num. selected variables (mean)', color='navy')  
ax2.plot(alpha, NumVarSelectMean, alpha=0.8, color='navy')
ax2.plot(alpha_min, NumVarSelect_min,'o', color='navy')
ax2.text(alpha_min, NumVarSelect_min,'   3 vars', color='navy')
ax2.tick_params(axis='y', labelcolor='navy')
fig.tight_layout()  
plt.show()

"""
    Analysis of the stability of the selected variables.
    Shows graphically which variables are selected in each iteration, 
    for the alpha value that produces the minimum eSplit.
"""

plt.figure()

for i in range(len(listColNames)):
    plt.axhline(y=i, color='gray', linewidth=0.1)
    
for j in range(66):
    selVars = np.where(Beta[np.where(eSplitMean == eSplit_min)[0].tolist()[0]][j] != 0)
    plt.plot(np.ones(np.size(selVars))*j, selVars[0], 'o', color=(0.1, 0.2+0.01*j, 0.15+0.012*j), alpha=0.9-0.007*j)
    
plt.title('Selected Variables in each iteration of LOO (eSplit-min)')
plt.xlabel('i-th Leave-one-out iteration')
plt.yticks(np.arange(len(listColNames)), listColNames)
plt.text(40,2,'Mean eSplit = ' + str(round(eSplit_min,3)), color='gray');


"""
    Reads and prepares the new data frame
"""

dataframe=pandas.read_csv("./MedicalData2.csv",sep=',',decimal=b'.')

listColNames=list(dataframe.columns)

XY=dataframe.values
ColNb_Y=listColNames.index('Pathology type')

#reshape is to make sure that Y is a column vector
Y=XY[:,ColNb_Y].reshape((XY.shape[0],1))   
X = np.delete(XY, ColNb_Y, 1)

X_scaled = preprocessing.scale(X)

#to make it contain the column names of X only
listColNames.pop(ColNb_Y);     

# Find the probability of each pathology type for each observation from 1 ... to n=66
logReg = LogisticRegression(penalty='l2')
logReg.fit(X_scaled,Y.reshape(n_lines,))
prob = logReg.predict_proba(X_scaled)

# Plots the probability
width = 0.6
plt.bar(np.arange(0,66), prob[:,0], width, color='navy')
plt.bar(np.arange(0,66), prob[:,1], width, bottom=prob[:,0], color='orange')
plt.title('Probability of Pathology Type being 1 or 2')
orange_patch = mpatches.Patch(color='orange', label='Pathology type 2')
blue_patch = mpatches.Patch(color='navy', label='Pathology type 1')
plt.legend(handles=[orange_patch, blue_patch], loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Observations')
plt.ylabel('Probability')
plt.show()

"""
    Plots the predictions and calculates the prediction 
    error to implement color-mapping (red and green) to 
    distinguish between good and bad predictions.
"""
# calculates the error in each prediction
err = np.abs(np.subtract(logReg.predict(X_scaled).reshape(-1,1), Y))

# marker size with a radius proportional to the prediction probability
s = []
for i in range(n_lines):
    if prob[i,1] >= 0.5:
        s.append(60*prob[i,1])
    else:
        s.append(60*prob[i,0])

# defines a color vector, green when error = 0 and red when error = 1
color = ['green', 'red']
color = [color[int(i)] for i in err]

# Plots predictions with color mapping for correct and incorrect
plt.scatter(np.arange(66), logReg.predict(X_scaled), s=s, c=color, alpha=0.6)
plt.xlabel('Observations')
plt.ylabel('Pathology type prediction')
red_patch = mpatches.Patch(color='red', label='Incorrect prediction')
blue_patch = mpatches.Patch(color='green', label='Correct prediction')
plt.legend(handles=[red_patch, blue_patch]);

numErrors = np.sum(err == 1)
errorRate = (numErrors / n_lines)
print('> Logistic Regression prediction completed with %d errors.' % numErrors)
print('> Corresponds to an error rate of %.2f.' % errorRate)

"""
    Performs a logistic regression using L1 penalty
    and plots the predicted probability of the 
    pathology being type 1 or 2.
"""

# Logistic Regression for multiple penalty factors. The lower the alpha, the greater the regularisation
alpha = [0.001+0.002*i for i in range(1, 100)]
[eSplit, NumVarSelect, Beta] = optimizeSelection(X_scaled,Y.reshape(n_lines,), alpha, method='log')

"""
    Calculates the average eSplit for each alpha and the 
    average number of selected variables for each alpha.
"""

eSplitMean = np.mean(eSplit, axis=1)
NumVarSelectMean = np.round(np.mean(NumVarSelect, axis=1))

"""
    Finds the minimum eSplit that verifies number of selected variables = 1,
    the corresponding alpha and the actual number of selected variables.
"""

# find the minimum eSplit with max 1 vars selected
eSplit_min = np.min(eSplitMean[np.where(NumVarSelectMean==1)])

# find the alpha that generates the lowest eSplit for 1 var selected
alpha_min = alpha[np.where(eSplitMean == eSplit_min)[0].tolist()[0]]

# find the number of variables satisfying: number of selected vars = 1 and lowest eSplit
NumVarSelect_min = NumVarSelectMean[np.where(eSplitMean == eSplit_min)[0].tolist()[0]]

"""
    Shows the results of the optimization graphically.
"""

fig, ax1 = plt.subplots()
plt.axvline(x=alpha_min, color='lightgray', linestyle='--')
ax1.axhline(y=eSplit_min, color='lightgray', linestyle='--')
ax1.set_title('eSplit and Num. selected variables vs. alpha (LOO)')
ax1.set_xlabel('alpha (LogReg)')
ax1.set_ylabel('Mean eSplit', color='green')
ax1.plot(alpha, eSplitMean, alpha=0.8, color='green')
ax1.plot(alpha_min, eSplit_min,'o', color='green')
ax1.tick_params(axis='y', labelcolor='green')
ax1.text(alpha_min, eSplit_min,'  eSplit min', color='green')
ax2 = ax1.twinx()
ax2.axhline(y=NumVarSelect_min, color='lightgray', linestyle='--')
ax2.set_ylabel('Num. selected variables (mean)', color='navy')  
ax2.plot(alpha, NumVarSelectMean, alpha=0.8, color='navy')
ax2.plot(alpha_min, NumVarSelect_min,'o', color='navy')
ax2.text(alpha_min, NumVarSelect_min,'   1 var', color='navy')
ax2.tick_params(axis='y', labelcolor='navy')
fig.tight_layout()  
plt.show()

"""
    Analysis of the stability of the selected variables.
    Shows graphically which variables are selected in each iteration, 
    for the alpha value that produces the minimum eSplit.
"""

plt.figure()

for i in range(len(listColNames)):
    plt.axhline(y=i, color='gray', linewidth=0.1)
    
for j in range(66):
    selVars = np.where(Beta[np.where(eSplitMean == eSplit_min)[0].tolist()[0]][j] != 0)
    plt.plot(np.ones(np.size(selVars))*j, selVars[0], 'o', color=(0.1, 0.2+0.01*j, 0.15+0.012*j), alpha=0.9-0.007*j)
    
plt.title('Selected Variables in each iteration of LOO (eSplit-min)')
plt.xlabel('i-th Leave-one-out iteration')
plt.yticks(np.arange(len(listColNames)), listColNames)
plt.text(40,2,'Mean eSplit = ' + str(round(eSplit_min,3)), color='gray');

"""
    Plots every variable 1...18 with color mapping to Pathology 
    type 1 and Pathology type 2 to identify if Cells 2 density 
    is in fact linked to the pathology type.
"""

f, ax = plt.subplots(nrows = 6, ncols = 3)
f.tight_layout()

green_patch = mpatches.Patch(color='green', label='Pathology Type 1')
blue_patch = mpatches.Patch(color='navy', label='Pathology Type 2')
plt.legend(handles=[green_patch, blue_patch], loc='center left', bbox_to_anchor=(1.2, 0.5));

for index in range(n_col):
    ColNb_Var = index
    c = []
    for i in range(n_lines):
        if Y[i] == 1:
            c.append('navy')
        else:
            c.append('green')
    j = int(ColNb_Var % 3)
    i = int(np.trunc(ColNb_Var / 3))
    ax[i][j].scatter(X[:,ColNb_Var].reshape(1,-1), np.zeros_like(X[:,ColNb_Var].reshape(1,-1)), c=c, alpha=0.6)
    ax[i][j].set_xlabel(listColNames[ColNb_Var])
    ax[i][j].get_yaxis().set_ticks([])
    
"""
    Uses just 'Cells 2 density' to 
    perform the logistic regression.
"""

logReg = LogisticRegression()
logReg.fit(X_scaled[:,listColNames.index('Cells 2 density')].reshape(-1, 1),Y.reshape(n_lines,))
R = logReg.score(X_scaled[:,listColNames.index('Cells 2 density')].reshape(-1, 1),Y.reshape(n_lines,))
print('> Fit score = %.2f' % round(R,2))

"""
    Plots the predictions and calculates the prediction 
    error to implement color-mapping (red and green) to 
    distringuish between good and bad predictions.
"""
# calculates the error in each prediction
err = np.abs(np.subtract(logReg.predict(X_scaled[:,listColNames.index('Cells 2 density')].reshape(-1, 1)), Y[:,0]))
print(type(logReg.predict(X_scaled[:,listColNames.index('Cells 2 density')].reshape(-1, 1))))

# marker size with a radius proportional to the prediction probability
s = []
for i in range(n_lines):
    if prob[i,1] >= 0.5:
        s.append(60*prob[i,1])
    else:
        s.append(60*prob[i,0])

# defines a color vector, green when error = 0 and red when error = 1
color = ['green', 'red']
color = [color[int(i)] for i in err]

# Plots predictions with color mapping for correct and incorrect
plt.scatter(np.arange(66), logReg.predict(X_scaled[:,listColNames.index('Cells 2 density')].reshape(-1, 1)), s=s, c=color, alpha=0.6)
plt.xlabel('Observations')
plt.ylabel('Pathology type prediction')
plt.title('Pathology type prediction using Cells 2 density')
red_patch = mpatches.Patch(color='red', label='Incorrect prediction')
green_patch = mpatches.Patch(color='green', label='Correct prediction')
plt.legend(handles=[red_patch, green_patch]);
