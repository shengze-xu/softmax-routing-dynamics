#!/usr/bin/env python3
"""Numerical checks for the full-hull case and exponents above one."""
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import softmax

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.studies.mobility_exponent_continuum import integrate, write_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/tables')
    args=parser.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    rows=[integrate(n,eps,alpha,max_step=step) for n in (4,8)
          for alpha in (1.5,2.0) for eps in (1e-9,1e-8,1e-7,1e-6,1e-5)
          for step in (.5,.25)]
    assert max(r['relative_time_error'] for r in rows)<1e-9
    write_rows(out/'recovery_exponents_above_one.csv',rows)
    for alpha in (1.5,2.):
        group=[r for r in rows if r['n_experts']==4 and r['alpha']==alpha and r['max_step_s']==.25]
        slope=-np.polyfit(np.log([r['epsilon'] for r in group]),np.log([r['observed_time'] for r in group]),1)[0]
        assert abs(slope-alpha)<1e-4
        print('alpha',alpha,'fitted',slope)
    print('maximum quadrature error',max(r['relative_time_error'] for r in rows))
    # y lies in the relative interior of the expert triangle. Unique barycentric
    # representation and analytic proper-face stationary losses give a direct check.
    E=np.array([[0.,0.],[1.,0.],[0.,1.]])
    y=np.array([.3,.25]); pstar=np.array([.45,.3,.25])
    ell_comp=min(y[0]**2,y[1]**2,(1-y.sum())**2/2)
    q0=np.log([.55,.24,.21])
    loss0=np.sum((E.T@softmax(q0)-y)**2)
    assert loss0<ell_comp
    def rhs(t,q):
        p=softmax(q);J=np.diag(p)-np.outer(p,p)
        return -J@E@(E.T@p-y)
    errors=[]
    for tol in (1e-10,1e-12):
        sol=solve_ivp(rhs,[0,2000],q0,method='DOP853',rtol=tol,atol=tol*.01)
        assert sol.success
        err=np.linalg.norm(softmax(sol.y[:,-1])-pstar)
        assert err<1e-8
        errors.append(dict(relative_tolerance=tol,initial_loss=loss0,
                           lowest_suboptimal_stationary_loss=ell_comp,
                           terminal_probability_error=float(err)))
    write_rows(out/'full_hull_projection_check.csv',errors)
    print('full-hull probability errors',[r['terminal_probability_error'] for r in errors])


if __name__=='__main__':main()
