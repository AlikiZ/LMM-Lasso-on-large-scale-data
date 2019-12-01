import sys
import os
import argparse
import logging
import numpy

#the following 3 lines are my modification in case the cause problem -> delete
#right now it is not working without them
import sys
import os
sys.path.append(os.path.abspath("/home/aliki/Documents/hpi/llmlasso/AdaScreen"))

from clustermap import Job, process_jobs
import adascreen.solver #as solver
from sklearn import linear_model
from experiment_impl import *

from adascreen.screening_rules import EDPP, DOME, DPP, SAFE, StrongRule, HSConstr
from adascreen.bagscreen import BagScreen
from adascreen.adascreen import AdaScreen
from adascreen.sasvi import Sasvi

#check for validity of importing cython .pyx file
import pyximport
pyximport.install()
import scripts.example_cy as example
#import adascreen.enet_solver as enet_solver    ImportError: Building module adascreen.enet_solver failed: ['ImportError: /home/aliki/.pyxbld/lib.linux-x86_64-2.7/adascreen/enet_solver.so: undefined symbol: cblas_dasum\n']

#print(example.test(3))


x = numpy.array([[0,0], [1, 1], [2, 2]])
y = numpy.array([0, 1, 2])
#define an easy example and solve it using sklearn linear_model
clf = linear_model.Lasso(alpha=0.1)
clf.fit(x, y)

print "the linear model Lasso coef: {0}".format(clf.coef_)
print "the linear model Lasso intercept: {0}".format(clf.intercept_)

#use the solver from adascreen.solver
#mdl1 = adascreen.solver.SklearnLarsSolver()
#(a,b,c) = mdl1.solve(start_pos=0, X=x, y=y, l1_reg=1.0,l2_reg=0.0)
# print a

#def solve(self, start_pos, X, y, l1_reg, l2_reg, max_iter=20000, tol=1e-6):
#works: SklearnLarsSolver,
#does not work: SklearnCDSolver, rest...

#try to do it like in the test_lasso.py
EXPERIMENT_LIST = [ \
        screening_performance_one_shot,\
        screening_performance_sequential,\
        path_times, \
        path_speed_up, \
        path_solver_acceleration, \
        path_solver_acceleration_nopath, \
        path_accuracy, \
        ]
numpy.random.seed(int(32))

ada_hs = AdaScreen(EDPP())
ada_hs.add_local_hs_rule(HSConstr(max_constr=100))

ada_sasvi = AdaScreen(EDPP())
ada_sasvi.add_local_hs_rule(Sasvi())

ada_sasvi1 = AdaScreen(Sasvi())
ada_sasvi1.add_local_hs_rule(Sasvi())

ada_dome = AdaScreen(EDPP())
ada_dome.add_local_hs_rule(DOME())

ada_full = AdaScreen(EDPP())
ada_full.add_local_hs_rule(Sasvi())
ada_full.add_local_hs_rule(HSConstr(max_constr=100))

ada_strong = AdaScreen(StrongRule())
ada_strong.add_local_hs_rule(Sasvi())
ada_strong.add_local_hs_rule(HSConstr(max_constr=10))

bag = BagScreen(EDPP())
bag.add_rule(DOME())
#bag.add_rule(Sasvi())

ada_bag = AdaScreen(EDPP())
ada_bag.add_global_hs_rule(DOME())
#ada_bag.add_local_hs_rule(Sasvi())

ada_dome1 = AdaScreen(DOME())
ada_dome1.add_global_hs_rule(DOME())

ada1 = AdaScreen(EDPP())
ada1.add_local_hs_rule(HSConstr(max_constr=1))
ada2 = AdaScreen(EDPP())
ada2.add_local_hs_rule(HSConstr(max_constr=5))
ada3 = AdaScreen(EDPP())
ada3.add_local_hs_rule(HSConstr(max_constr=10))
ada4 = AdaScreen(EDPP())
ada4.add_local_hs_rule(HSConstr(max_constr=100))
ada5 = AdaScreen(EDPP())
ada5.add_local_hs_rule(HSConstr(max_constr=1000))

ruleset_all = [ada_full, ada_hs, ada_sasvi, Sasvi(), EDPP(), DPP(), DOME(), SAFE()]
screening_rules = eval('ruleset_{0}'.format('all'))
# optionally overwrite the global screening_rules by a single rule
screening_rules = [screening_rules[1]]
use_solver = 1
(x, res, props) = EXPERIMENT_LIST[6](x, y, solver_ind=use_solver, screening_rules=screening_rules,
                                                        steps=65, geomul=0.9)

print res
#error: expected 'double' but got 'long'
#change the solver: experiment_impl.py -> line 178     myLasso = ScreeningLassoPath(ScreenDummy(), solver[0], path_lb=lower_bound, path_steps=steps, path_stepsize=geomul, path_scale='geometric')


# w2 = train_lasso_adascreen(SUX, SUy, mu)         #linear_model.Lasso
# w3 = train_lasso_3(SUX, SUy, mu)
# w4 = train_lasso_4(SUX, SUy, mu)

def train_lasso_2(X,y,mu):
    model = linear_model.Lasso(alpha=mu, fit_intercept=False)
    model.fit(X * NP.sqrt(X.shape[0]), y * NP.sqrt(X.shape[0]))

    w = model.coef_
    return w

def train_lasso_3(X,y,mu):
    model = linear_model.LassoLars(alpha=mu, fit_intercept=False)
    model.fit(X * NP.sqrt(X.shape[0]), y * NP.sqrt(X.shape[0]), Xy=None)
    return model.coef_

def train_lasso_4(X,y,mu):
    model = glmnet( x = X * NP.sqrt(X.shape[0]), y = y * NP.sqrt(X.shape[0]), family = 'gaussian', alpha = 1, nlambda = 20)
    w = glmnetCoef(model)
    #model.fit((X * NP.sqrt(X.shape[0]), y * NP.sqrt(X.shape[0]))
    #w = model.coef_
    return w

def train_lasso_adascreen(X,y,mu):
    EXPERIMENT_LIST = [ \
        screening_performance_one_shot, \
        screening_performance_sequential, \
        path_times, \
        path_speed_up, \
        path_solver_acceleration, \
        path_solver_acceleration_nopath, \
        path_accuracy, \
        ]
    NP.random.seed(int(32))

    ada_hs = AdaScreen(EDPP())
    ada_hs.add_local_hs_rule(HSConstr(max_constr=100))

    ada_sasvi = AdaScreen(EDPP())
    ada_sasvi.add_local_hs_rule(Sasvi())

    ada_sasvi1 = AdaScreen(Sasvi())
    ada_sasvi1.add_local_hs_rule(Sasvi())

    ada_dome = AdaScreen(EDPP())
    ada_dome.add_local_hs_rule(DOME())

    ada_full = AdaScreen(EDPP())
    ada_full.add_local_hs_rule(Sasvi())
    ada_full.add_local_hs_rule(HSConstr(max_constr=100))

    ada_strong = AdaScreen(StrongRule())
    ada_strong.add_local_hs_rule(Sasvi())
    ada_strong.add_local_hs_rule(HSConstr(max_constr=10))

    bag = BagScreen(EDPP())
    bag.add_rule(DOME())
    # bag.add_rule(Sasvi())

    ada_bag = AdaScreen(EDPP())
    ada_bag.add_global_hs_rule(DOME())
    # ada_bag.add_local_hs_rule(Sasvi())

    ada_dome1 = AdaScreen(DOME())
    ada_dome1.add_global_hs_rule(DOME())

    ada1 = AdaScreen(EDPP())
    ada1.add_local_hs_rule(HSConstr(max_constr=1))
    ada2 = AdaScreen(EDPP())
    ada2.add_local_hs_rule(HSConstr(max_constr=5))
    ada3 = AdaScreen(EDPP())
    ada3.add_local_hs_rule(HSConstr(max_constr=10))
    ada4 = AdaScreen(EDPP())
    ada4.add_local_hs_rule(HSConstr(max_constr=100))
    ada5 = AdaScreen(EDPP())
    ada5.add_local_hs_rule(HSConstr(max_constr=1000))

    adatry = AdaScreen(EDPP())
    adatry.add_local_hs_rule(ScreenDummy())

    ruleset_all = [ada_full, ada_hs, ada_sasvi, Sasvi(), EDPP(), DPP(), DOME(), SAFE()]
    screening_rules = eval('ruleset_{0}'.format('all'))
    # optionally overwrite the global screening_rules by a single rule
    screening_rules = [screening_rules[1]]
    use_solver = 1
    (x, res, props) = EXPERIMENT_LIST[6](X, y, solver_ind=use_solver, screening_rules=screening_rules,
                                         steps=65, geomul=0.9)

    #weights of Lasso
    return 0
