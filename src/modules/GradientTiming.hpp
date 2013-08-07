
/*
 *  codeare Copyright (C) 2007-2010 Kaveh Vahedipour
 *                               Forschungszentrum Juelich, Germany
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful, but 
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
 *  General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 
 *  02110-1301  USA
 */

#ifndef __GRADIENT_TIMING_H__
#define __GRADIENT_TIMING_H__

#include "Algos.hpp"
#include "CX.hpp"
#include "Math.hpp"
#include "Interpolate.hpp"
#include "IOContext.hpp"

using namespace codeare::matrix::io;

#define announce_plan

struct Solution {
	
	Matrix<double> k;
	Matrix<double> g;
	Matrix<double> s;
	Matrix<double> t;

    void dump (IOContext& f) const {
        fwrite (f, g);
        fwrite (f, k);
        fwrite (f, s);
        fwrite (f, t);
    }
	
};


struct GradientParams {
	
	Matrix<double> k;    /**< K-Space trajectory [1/cm] */
	double         mgr;  /**< Maximum allowed gradient */
	double         msr;  /**< Maximum allowed slewrate */
	double         dt;   /**< Sampling interval */
	int            gunits;
	int            lunits;
	
};


inline static double
RungeKutta (const double s, const double   ds, const double st, 
			const double* k, const double smax, const double  L, 
			const bool  fw) {

	double k1, k2, k3, k4;
	size_t l = (fw) ? 0 : 2, m = 1,  n = (fw) ? 2 : 0;
	double pgs = pow (GAMMA / 10.0 * smax, 2);

	k1 = ds * sqrt (pgs - pow(k[l],2) * pow (st       ,3));
	k2 = ds * sqrt (pgs - pow(k[m],2) * pow (st + k1/2,3));
	k3 = ds * sqrt (pgs - pow(k[m],2) * pow (st + k2/2,3));
	k4 = ds * sqrt (pgs - pow(k[n],2) * pow (st + k3/2,3));
	
	return k1/6 + k2/3 + k3/3 + k4/6;
	
}


struct SDIn {

	PolyVal<double> *pkx, *pky, *pkz;
	Matrix<double> posh, sh;
	double mgr, msr;

	~SDIn () {
		
		delete pkx;
		delete pky;
		delete pkz;
		
	}

};


struct SDOut {

	Matrix<double> k;
	Matrix<double> phi;

    void dump (IOContext& f) {
        fwrite (f,k);
        fwrite (f,phi);
    }

};



SDOut SDMax (SDIn& si) {

	SDOut so;

	Matrix<double>& posh = si.posh;
	Matrix<double>& sh   = si.sh;

	PolyVal<double>& pkx = *(si.pkx);
	PolyVal<double>& pky = *(si.pky);
	PolyVal<double>& pkz = *(si.pkz);

	size_t ssp = size(posh,0);
	size_t sss = size(sh  ,0);

	Matrix<double>  dpp (ssp,1);
	Matrix<double>  dsp (sss,1);
	Matrix<double>  csp (ssp,3);

	for (size_t i = 0; i < ssp-1; i++)
		dpp[i] = posh[i+1] - posh[i]; 
	dpp[ssp-1] = dpp[ssp-2];

	for (size_t i = 0; i < sss-1; i++)
		dsp[i] = sh[i+1] - sh[i]; 
	dsp[ssp-1] = dsp[ssp-2];
	
	printf ("."); fflush (stdout);

	for (size_t i = 1; i < ssp-1; i++) {
		csp(i,0) = (pkx.Lookup (posh[i] + dpp[i]) - pkx.Lookup (posh[i])) / dsp[i];
		csp(i,1) = (pky.Lookup (posh[i] + dpp[i]) - pky.Lookup (posh[i])) / dsp[i];
		csp(i,2) = (pkz.Lookup (posh[i] + dpp[i]) - pkz.Lookup (posh[i])) / dsp[i];
	}

	printf ("."); fflush (stdout);

	dpp.Clear();

	Matrix<double>  dpm (ssp,1);
	Matrix<double>  dsm (sss,1);
	Matrix<double>  csm (ssp,3);

	for (size_t i = 1; i < ssp;   i++) 
		dpm[i] = posh[i] - posh[i-1]; 
	dpm[0] = dpm[1];

	for (size_t i = 1; i < sss;   i++) 
		dsm[i] = sh[i] - sh[i-1]; 
	dsm[0] = dsm[1];

	for (size_t i = 1; i < ssp-1; i++) {
		csm(i,0) = (pkx.Lookup (posh[i]) - pkx.Lookup (posh[i] - dpm[i])) / dsm[i];
		csm(i,1) = (pky.Lookup (posh[i]) - pky.Lookup (posh[i] - dpm[i])) / dsm[i];
		csm(i,2) = (pkz.Lookup (posh[i]) - pkz.Lookup (posh[i] - dpm[i])) / dsm[i];
	}

	dpm.Clear();
	
	Matrix<double>  css (ssp,3);

	for (size_t i = 1; i < ssp-1; i++) {
		css(i,0) = (csp(i,0) - csm(i,0)) / ((dsm[i] + dsp[i]) * 0.5);
		css(i,1) = (csp(i,1) - csm(i,1)) / ((dsm[i] + dsp[i]) * 0.5);
		css(i,2) = (csp(i,2) - csm(i,2)) / ((dsm[i] + dsp[i]) * 0.5);
	}

	csp.Clear();
	csm.Clear();
	dsp.Clear();
	dsm.Clear();
	
	printf (". "); fflush (stdout);

	so.k   = Matrix<double> (ssp,1);
	so.phi = Matrix<double> (ssp,1);

	for (size_t i = 1; i < ssp-1; i++) 
		so.k[i]  = sqrt (pow(css(i,0),2) + pow(css(i,1),2) + pow(css(i,2),2));
	
	so.k[0]     = so.k[1]; 
	so.k[sss-1] = so.k[sss-2];
	
	css.Clear();

	double mgr = GAMMA / 10.0 * si.mgr;
	double msr = GAMMA / 10.0 * si.msr;

	for (size_t i = 0; i < ssp; i++) 
		so.phi[i] = MIN (mgr, sqrt(msr/so.k[i]));
	
	return so;

}



/**
 * @brief       Compute real world k-space / gradient solution with hardware limits
 *
 * @param  gp   Parameters
 * @return      Solution
 */
Solution ComputeGradient (GradientParams& gp) {

	Solution s;
	size_t ups = 25, ts = 0;

	printf ("  Const arc-length parametrization "); fflush(stdout);
	ticks start = getticks();

	Matrix<double> op, np, sop;

	size_t sgpk = size(gp.k,0);
	size_t ssk  = (sgpk-1)*ups+1;

	op = linspace<double> (0.0, (double)(size(gp.k,0)-1), sgpk);
	np = linspace<double> (0.0, (double)(size(gp.k,0)-1),  ssk);

	printf ("..."); fflush (stdout);

	s.k = interp1 (op, gp.k, np);

	op.Clear();

	size_t snp = size(s.k,0);
	s.g = Matrix<double>(snp,3);
	sop = Matrix<double>(snp,1);

	SDIn sdin;

	sdin.pkx = new PolyVal<double> (np, (double*)&(s.k)[0]);
	sdin.pky = new PolyVal<double> (np, (double*)&(s.k)[1*snp]);
	sdin.pkz = new PolyVal<double> (np, (double*)&(s.k)[2*snp]);

	printf ("..."); fflush (stdout);

	sdin.mgr = gp.mgr;
	sdin.msr = gp.msr;

	for (size_t i = 0, j = snp, l = 2*snp; i < snp-1; i++) {
		s.g[i  ] = s.k[i+1  ] - s.k[i  ];
		s.g[i+j] = s.k[i+1+j] - s.k[i+j];
		s.g[i+l] = s.k[i+1+l] - s.k[i+l];
	}

	printf (".."); fflush (stdout);

	s.g *= (double)ups;
	
	sop [0] = 0.0;
	
	for (size_t i = 0; i < snp-1; i++)
		sop [i+1] = sop[i] + sqrt (pow(s.g(i,0),2) + pow(s.g(i,1),2) + pow(s.g(i,2),2));

	sop /= (double)ups;

	printf ("."); fflush (stdout);
	
	double st0 = GAMMA / 10.0 * gp.msr * gp.dt;
	double ds  = st0 * gp.dt / 3.0;
	
	double L   = sop(numel(sop)-1);
	ts = ceil(L/ds);

	Matrix<double> sf, sta, stb, pos;
	
	printf ("."); fflush (stdout);
	
	sf         = linspace<double> (0.0, L,   ts);
	sdin.sh    = linspace<double> (0.0, L, 2*ts);
	sta        = zeros<double> (size(sf,0), 1);
	stb        = zeros<double> (size(sf,0), 1);

	sdin.posh  = interp1 (sop, np, sdin.sh);
	
	np.Clear();
	sop.Clear();

	printf ("."); fflush (stdout);

	pos        = Matrix<double> (size(sdin.posh,0)/2,1);

	for (size_t i = 0; i < size(pos,0); i++)
		pos[i] = sdin.posh[i*2];

	printf (" done: (%.3f)\n  Computing geometry dependent constraints ", elapsed(getticks(), start) / Toolbox::Instance()->ClockRate()); 
	fflush(stdout);
	start = getticks();

	SDOut sdout = SDMax (sdin);
	size_t sk = size(sdout.k,0);
	size_t ss = size(sf,0);

	sdout.k = resize (sdout.k,sk+2,1);
	sdout.k[sk]   = sdout.k[sk-1];
	sdout.k[sk+1] = sdout.k[sk-1];

	sta[0] = 0.0;

    //OK

	printf ("done: (%.3f)\n  Solving ODE forward ........................ ", elapsed(getticks(), start) / Toolbox::Instance()->ClockRate());
	fflush(stdout);
	start = getticks();

	for (size_t i = 1; i < ss; i++)
		sta[i] = MIN (sta[i-1] + RungeKutta (sf[i], ds, sta[i-1], &sdout.k[(i-1)*2], gp.msr, L,  true), sdout.phi[i*2-1]);

	stb[ss-1] = sta[ss-1];

	printf ("done: (%.3f)\n  Solving ODE backward ....................... ", elapsed(getticks(), start) / Toolbox::Instance()->ClockRate()); 
	fflush(stdout);
	start = getticks();

	for (size_t i = ss-2; i > 0; i--)
		stb[i] = MIN (stb[i+1] + RungeKutta (sf[i], ds, stb[i+1], &sdout.k[(i+2)*2], gp.msr, L, false), sdout.phi[i*2-1]);

	printf ("done: (%.3f)\n  Final interpolations ....................... ", elapsed(getticks(), start) / Toolbox::Instance()->ClockRate());
	fflush(stdout);

	start = getticks();

	for (size_t i = 0; i < ss; i++)
		sta[i] = (sta[i] <= stb[i]) ? sta[i] : stb[i];

	Matrix<double> tos, sot, t, pot;
	double T;
	size_t Nt;
    
	tos  = ds/sta;
    tos[0] = 0;
    tos = cumsum (tos);

	T      = tos [numel(tos)-1];
	Nt     = round (T/gp.dt); 

	t      = linspace<double> (0.0, T, Nt);

	sot    = interp1 (tos,  sf,   t);
	pot    = interp1 ( sf, pos, sot);

	pos.Clear();
	
	gp.k   = Matrix<double> (Nt,3);

	for (size_t i = 0; i < Nt; i++) {
		gp.k(i,0) = (sdin.pkx->Lookup (pot[i]));
		gp.k(i,1) = (sdin.pky->Lookup (pot[i]));
		gp.k(i,2) = (sdin.pkz->Lookup (pot[i]));
	}

	s.g    = Matrix<double> (Nt-1,3);
	s.k    = Matrix<double> (Nt-1,3);
	s.s    = Matrix<double> (Nt-1,3);
	
	for (size_t i = 0; i < Nt-1; i++) {
		s.g(i,0) = (gp.k(i+1,0) - gp.k(i,0)) / (GAMMA / 10.0 * gp.dt);
		s.g(i,1) = (gp.k(i+1,1) - gp.k(i,1)) / (GAMMA / 10.0 * gp.dt);
		s.g(i,2) = (gp.k(i+1,2) - gp.k(i,2)) / (GAMMA / 10.0 * gp.dt);
	}

	for (size_t i = 1; i < Nt-1; i++) {
		s.k(i,0) =  s.k(i-1,0) + s.g(i,0) * (GAMMA / 10.0 * gp.dt);
		s.k(i,1) =  s.k(i-1,1) + s.g(i,1) * (GAMMA / 10.0 * gp.dt);
		s.k(i,2) =  s.k(i-1,2) + s.g(i,2) * (GAMMA / 10.0 * gp.dt);
	}

	for (size_t i = 0; i < 3; i++)
		s.k(0,i) = 0.0;

	for (size_t i = 0; i < Nt-2; i++) {
		s.s(i,0) = (s.g(i+1,0) - s.g(i,0)) / gp.dt;
		s.s(i,1) = (s.g(i+1,1) - s.g(i,1)) / gp.dt;
		s.s(i,2) = (s.g(i+1,2) - s.g(i,2)) / gp.dt;
	}

	for (size_t i = 0; i < 3; i++)
		s.s(Nt-2,i) = s.s(Nt-3,i);  

	t = resize (t, Nt-1, 1);
	s.t = t;

	printf ("done: (%.3f)\n", elapsed(getticks(), start) / Toolbox::Instance()->ClockRate());

	return s;
	
}


#endif /* __GRADIENT_TIMING_H__ */
