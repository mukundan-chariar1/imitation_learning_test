import jax.numpy as jp

from pdb import set_trace as st
from jax.debug import breakpoint as jst

def expm(mat):
    eigvalue, eigvectors=jp.linalg.eigh(mat)
    e_Lambda=jp.eye(jp.size(mat, 0))*(jp.exp(eigvalue))
    e=eigvectors*e_Lambda*eigvectors.I

    return e