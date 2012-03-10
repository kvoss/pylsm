#!/usr/bin/python

__author__ = 'Krzysztof Voss'
__version__ = '0.1'

import numpy as np
import scipy as sp
from numpy.core.getlimits import finfo
from scipy import ndimage

import matplotlib.pyplot as plt
import mahotas

#import unittest
import ConfigParser
import logging
import os

##### Importing constants
config = ConfigParser.ConfigParser()
config.read('lsmethod.cfg')

which_H_for_M   = config.get('LSM', 'which_H_for_M')
init_sdf_mthd   = config.get('LSM', 'init_sdf_mthd')

use_dbl_pic     = config.getboolean('LSM', 'use_dbl_pic')
grad_ini_fi     = config.getboolean('LSM', 'grad_ini_fi')

use_pythode     = config.getboolean('LSM', 'use_pythode')
make_movie      = config.getboolean('LSM', 'make_movie')

initial_ls_val  = config.getfloat('LSM', 'initial_ls_value')
ls_threshold    = config.getfloat('LSM', 'ls_threshold')

sigma           = config.getfloat('LSM', 'sigma')
n               = config.getint('LSM', 'n')

epsilon     = config.getfloat('LSM', 'epsilon')
lambda1     = config.getfloat('LSM', 'lambda1')
lambda2     = config.getfloat('LSM', 'lambda2')
upsilon     = config.getfloat('LSM', 'upsilon')
mu          = config.getfloat('LSM', 'mu')
eta         = config.getfloat('LSM', 'eta')

delta_t     = config.getfloat('LSM', 'delta_t')
t_0         = config.getfloat('LSM', 't_0')
t_f         = config.getfloat('LSM', 't_f')

R_min       = config.getfloat('LSM', 'R_min')
R_max       = config.getfloat('LSM', 'R_max')

eps = finfo(float).eps

if use_pythode:
    from pythode.ivp import ConstantIVPSolver, IVPSolverModule, return_rejected
    import pythode.ivp.parameters as para
    from pythode.ivp.schemes import feuler
    from pythode.lib import deep_copy_update, parametrizable
    TOLERANCE = delta_t #1e-2
#############

logging.basicConfig(filename='lsmethod.log')

log = logging.getLogger("lsmethod")
log.setLevel(logging.DEBUG)

#############

def calc_intensity_diff(R_c, I):
    """Computes distance between the image intensity
      and R_c as in eq:8
    """
    R_min, R_max = R_c
    Ihi = (I - R_max) * (I > R_max)
    Ilo = (R_min - I) * (R_min > I)
    return Ihi + Ilo

def Heaviside(x):
    """Smoothed Heaviside function as in eq:11
    """
    return (1 + 2/np.pi * np.arctan(x/np.pi)) / 2

def UnitStep(x):
    """Eq 1.12 from Fedkiw Osher book
    """
    return 1 if x > 0 else 0

def Dirac(x):
    """Smoothed Dirac function as in eq:12
    """
    return (epsilon / (epsilon**2 + x**2 + eps)) / np.pi

def Kernel():
    """Gaussian Kernel
    """
    a = np.zeros((n,n))
    a[n/2,n/2] = 1.
    a = ndimage.filters.gaussian_filter(a, sigma)   
    log.debug("Kernel shape:" + str(a.shape))
    return a

Me_1 = Heaviside if which_H_for_M == 'Heaviside' else UnitStep
def Me_2(fi):
    return 1-Me_1(fi)

def calc_fs(fi, img, K_img):
    common_num = ndimage.filters.gaussian_filter(Me_1(fi)*img, sigma)
    common_dnum = ndimage.filters.gaussian_filter(Me_1(fi), sigma)
    
    f1 = common_num/common_dnum
    
    log.debug("calc_fs f1: %f / %f" % ( np.sum(np.abs(common_num)), np.sum(np.abs(common_dnum))))
    log.debug("calc_fs f2: %f / %f" % (np.sum(np.abs(K_img-common_num)), np.sum(np.abs(1-common_dnum))))
    
    f2 = (K_img-common_num)/(1-common_dnum)
    
    return [f1,f2]

def WMSD(fi, img, kernel=None, K_img=None):
    """Weighted Mean Squared Difference 
    """
    f1,f2 = calc_fs(fi, img, K_img)
    kernel=Kernel(),
    def fun(m, img, n, idx):
        fn = np.reshape(m, (n,n))
        ntnst = img.flat[idx[0]]
        idx[0] += 1
        return np.sum(kernel * (ntnst - fn)**2)

    e1 = ndimage.filters.generic_filter(f1, fun, size=(n,n), extra_arguments=(img,n, [0]))
    e2 = ndimage.filters.generic_filter(f2, fun, size=(n,n), extra_arguments=(img,n, [0]))
    
    return [e1, e2] 

def initialize_distance_fi(img):
    if init_sdf_mthd == 'otsu':
        iimg = img * 255
        iimg = iimg.astype('uint8')
        thr = mahotas.otsu(iimg)
        timg = iimg * (iimg < thr)
    
        nucleusT = mahotas.otsu(timg)
        cytoplasm = timg > nucleusT
    
        tmp = np.ones_like(img) * -1. * initial_ls_val
        tmp[cytoplasm] = 1. * initial_ls_val
        
        if grad_ini_fi:
            gy,gx = np.gradient(tmp)
            gtmp = gy+gx
            tmp = tmp * (gtmp == 0.)
    elif init_sdf_mthd == 'rcluster':
        tmp = np.random.random(img.shape)
        tmp[tmp < 0.5] =  -1. * initial_ls_val
        tmp[tmp > -1.] =  1. * initial_ls_val
    else:
        tmp = np.random.random(img.shape)
    
    return tmp 

def f(fi, t, img, K_img, squared_eliminator):
    """Function to be integrated
    @param y a list consisting of fi array and img
    @param t time
    """
    log.info("f: t = %f" % t)

    fi = fi.reshape(img.shape)
    dirac_delta = Dirac(fi)

    e1, e2 = WMSD(fi, img, K_img=K_img)
    data_fitting = -1. * dirac_delta * (lambda1*e1 - lambda2*e2)

    grad_fi_y,grad_fi_x  = np.gradient(fi)
    grad_lenght = (grad_fi_y**2 + grad_fi_x**2)**(1./2)
    norm_grad_y, norm_grad_x = grad_fi_y/(grad_lenght+eps), grad_fi_x/(grad_lenght+eps) 
    _, ng_xx = np.gradient(norm_grad_x)
    ng_yy, _ = np.gradient(norm_grad_y)
    div_norm = ng_xx + ng_yy

    arc_smoothing = upsilon * dirac_delta * div_norm
    
    ls_regularization = mu * (ndimage.filters.laplace(fi) - div_norm)
    eliminator_factor = -1. * eta * dirac_delta * squared_eliminator
    
    fi_new = data_fitting + arc_smoothing + ls_regularization + eliminator_factor
    return fi_new.flatten()

if use_pythode:
    @parametrizable
    class FiFunction(object):
        def __init__(self,p):
            self.img = p['img']
            self.K_img = p['K_img']
            self.squared_eliminator = p['squared_eliminator']
    
        def __call__(self,t,y):
            return f(y, t, self.img, self.K_img, self.squared_eliminator)
        
    class WholeSolution(IVPSolverModule):
        def initialize(self,solver,solution):
            self.values = []
    
        @return_rejected
        def step(self,solver,solution):
            self.values.append((solution[0]['t'],solution[0][0]))
    
        def finalize(self,solver,solution):
            self.values.append((solution[0]['t'],solution[0][0]))
            solver['statistics']['whole solution'] = self.values        

    def pythode_solver(f, t0, t, args):
        img,K_img,squared_eliminator = args
    
        para.constant_solver_statistics['statistics modules'].append(WholeSolution)
        feuler_method_constant = deep_copy_update(para.solver,
                                                  para.constant_solver,
                                                  para.erk_solver,
                                                  {'integration module parameters':{'tableau':feuler}}, 
                                                  para.constant_solver_statistics)
        
        fifun_parameters = {'name':'Distance function evolution',
                            'initial time':t_0,
                            'initial values':t0,
                            'img': img,
                            'K_img':K_img,
                            'squared_eliminator':squared_eliminator,
                            'final time':t_f,
                            'final reference solution':t} 
    
        controls_constant = {'step size':delta_t}
        ivp = {'rhs':FiFunction,'ivp parameters':fifun_parameters}
        feuler_constant_solver_parameters = deep_copy_update(feuler_method_constant,ivp,controls_constant)
        
        feuler_constant_solver = ConstantIVPSolver(feuler_constant_solver_parameters)
        feuler_constant_solver.run()

        first_iter = True
        for s in feuler_constant_solver['statistics']['whole solution']:
            if first_iter:
                first_iter = False
                slns = s[1]
                continue
            slns = np.vstack((slns,s[1]))
        return slns

def simple_solver(f, t0, t, args):
    """Forward Euler solver
    """
    img,K_img,squared_eliminator = args
    fi = np.copy(t0)
    solns = np.array([t0])
    for tx in t:
        fi += delta_t * f(fi, tx,img,K_img,squared_eliminator)
        solns = np.vstack((solns,fi))
    return solns

def solve_ode(f, t0, t, args):
    """Calulating solutions to ODE
        t0 - initial vector (has to be .flatten())
    """
    if use_pythode:
        solver = pythode_solver
    else:
        solver = simple_solver
    slns = solver(f, t0, t, args)
    return slns

def segment(oimg):
    """ Segment nucleus and cytoplasm from picture
    """
    # blending color image
    if len(oimg.shape) == 3:
        img = np.sum(oimg, 2) / oimg.shape[2]
    else:
        img = oimg
    
    fi = initialize_distance_fi(img)
    t0 = np.array(fi).flatten()

    if not use_dbl_pic:
        img = img*255.0

    R_c = (R_min, R_max)
    K_img = ndimage.filters.gaussian_filter(img, sigma)

    t = np.arange(t_0, t_f, delta_t)
    squared_eliminator = (calc_intensity_diff(R_c, img))**2
    slns = solve_ode(f, t0, t, args=(img,K_img,squared_eliminator))
    return slns

def save_segmentation(fn):
    abs_fname = os.path.abspath(fn)
    dn, fname = os.path.split(abs_fname)
    log.info("segmenting file: " + fname)

    cellImg = plt.imread(abs_fname)
    slns = segment(cellImg)

    fname_tmp = '_'.join([fname,init_sdf_mthd,'ups'+str(upsilon),'T'+str(t_f),'eta'+str(eta),'eps'+str(epsilon)])
    os.mkdir(fname_tmp)

    for i in [ x for x in range(len(slns)) ]:
        # real image can have few channels, solution has two
        sol = slns[i,:].reshape(cellImg.shape[:2])
        
        blended_img = np.copy(cellImg)
        try:
            blended_img[np.abs(sol) < ls_threshold,0] = 1.
            
            blended_img[extract_nucleus(sol),0] = 1.
            blended_img[extract_cytoplasm(sol),1] = 1.
        except:
            blended_img[np.abs(sol) < ls_threshold] = 1.
            
            blended_img[extract_nucleus(sol)] = 1.
        
        imname=fname_tmp+'_'+str(i)+'.png'
        sp.misc.imsave(fname_tmp+'/'+imname, blended_img)

    if make_movie:    
        ifname = fname_tmp+'/'+fname_tmp + r"_%d.png"
        ofname = fname_tmp+'/'+fname_tmp + '.mp4'
        ffmpeg_cmd = 'ffmpeg -qscale 1 -r 6 -i ' + ifname + ' ' + ofname
        os.system(ffmpeg_cmd)

def segment_file(fname, only_final=True):
    """Conducts Level Set Method on given file

    Given file's name it opens it and returns final solution
    only_final whether to return only last solution of the whole set
    """
    afn = os.path.abspath(fname)
    img = plt.imread(afn)
    slns = segment(img)

    if only_final:
        ret = [ slns[-1].reshape(img.shape[:2]) ]
    else:
        ret = [ s.reshape(img.shape[:2]) for s in slns]
    return ret

def extract_cytoplasm(sln):
    """Given a solution from segmentation
        return a BW image corresponding to cytoplasm
    """
    return sln > 0.0

def extract_nucleus(sln):
    """Extracts nucleus from the solution
    """
    cp = extract_cytoplasm(sln)
    cell = ndimage.morphology.binary_fill_holes(cp)
    nucleus = cell - cp
    return nucleus

def calc_dice(bw1, bw2):
    """Calculates Dice coefficient of two BW images
    """
    dice_coeff = 2.*np.sum(np.logical_and(bw1,bw2)) / (np.sum(bw1) + np.sum(bw2))
    return dice_coeff

def calc_dice_cell(file_orig, file_truth):
    """Calculates dice coefficient of two files
            given by name regarding cytoplasm
            and nucleus similarity to file_truth

    Returns a tuple (cytoplasm, nucleus) displaying
    Dice's coefficient of these areas  
    for the whole evolution
    """
    # Ground Truth image with 0 and 1
    gt = plt.imread(file_truth)
    gtcyto = extract_cytoplasm(gt)
    gtnucl = extract_nucleus(gt)
    
    slns = segment_file(file_orig, only_final=False)
    dice_coeffs = []
    for sln in slns:
        cyto = extract_cytoplasm(sln)
        nucl = extract_nucleus(sln)
        
        cc = calc_dice(cyto, gtcyto) # cyto coeff
        nc = calc_dice(nucl, gtnucl) # nucl coeff
        dice_coeffs.append((cc, nc))
    return dice_coeffs

def usage():
    print """
    lsmethod.py [-h] [-p] [-n KERNEL] [-s SIGMA] [-d DELTA_T] [-t START_TIME] [-T END_TIME] [-u UPSILON] [-e ETA] [-r R_MIN] [-R R_MAX] [-E EPSILON] [-i INIT_METHOD] [-g GROUND_TRUTH_PATH] FILE

    --help
    -h      print help

    -p      use pythODE solver

    -n      kernel size
    -s      standard deviation for Gaussian kernel
    
    -d      step size
    -t      starting time
    -T      end time

    -u      smoothness coefficient

    -e      eliminator coefficient
    -r      lower mean intensity of cytoplasm
    -R      higher mean intensity of cytoplasm

    -E      Heaviside and Dirac functions' epsilon

    -i      initialization method: {otsu,rcluster,random}

    -g      ground truth file
            When given calculates Dice's coefficient of each evolution

    """

def main():
    import sys, getopt
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hpn:s:d:t:T:u:e:r:R:E:i:g:", ["help"])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(2)

    test_only = False
    global n, sigma, delta_t, t_0, t_f, upsilon, eta, R_min, R_max, epsilon, init_sdf_mthd, use_pythode
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-p",):
            use_pythode = True
        elif o in ("-n",):
            n = int(a)
        elif o in ("-s",):
            sigma = float(a)
        elif o in ("-d",):
            delta_t = float(a)
        elif o in ("-t",):
            t_0 = float(a)
        elif o in ("-T",):
            t_f = float(a)
        elif o in ("-u",):
            upsilon = float(a)
        elif o in ("-e",):
            eta = float(a)
        elif o in ("-r",):
            R_min = float(a)
        elif o in ("-R",):
            R_max = float(a)
        elif o in ("-E",):
            epsilon = float(a)
        elif o in ("-i",):
            init_sdf_mthd = a
        elif o in ("-g",):
            test_only = True
            gt_fname = a   
        else:
            assert False, "unhandled option"

    fname = args[0]
    if test_only:
        res = calc_dice_cell(fname, gt_fname)
        t_i = 0
        for d in res:
            log.info("%s %.2f %.2f %.2f %s"%(fname,t_0+t_i*delta_t,upsilon,eta,d))
            print fname,t_0+t_i*delta_t,upsilon,eta,d
            t_i += 1
    else:
        save_segmentation(fname)

if __name__  == "__main__":
    log.info("---> starting app")
    main()
    log.info("---> finishing app")

