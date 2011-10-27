/*
 *  jrrs Copyright (C) 2007-2010 Kaveh Vahedipour
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

#include "CompressedSensing.hpp"

using namespace RRStrategy;


RRSModule::error_code
CompressedSensing::Init () {

	printf ("Intialising CompressedSensing ...\n");

	for (size_t i = 0; i < 3; i++)
		m_N[i] = 1;

	int wli = 0;
	int m_fft = 0;

	Attribute ("dim",     &m_dim);
	Attribute ("Nx",      &m_N[0]);
	Attribute ("Ny",      &m_N[1]);
	Attribute ("Nz",      &m_N[2]);
    printf ("  Geometry: %iD (%i,%i,%i)\n", m_dim, m_N[0], m_N[1], m_N[2]);
	
	Attribute ("tvw",     &m_cgparam.tvw);
	Attribute ("xfmw",    &m_cgparam.xfmw);
    printf ("  Weights: TV(%.4f) L1(%.4f)\n", m_cgparam.tvw, m_cgparam.xfmw);
	
	Attribute ("fft",     &m_cgparam.fft);
	printf ("  FFT class: ");
	switch (m_fft) 
		{
		case 0:  printf ("%s", "Cartesian"); break;
		case 1:  printf ("%s", "Non-Cartesian"); break;
		default: printf ("%s", "Cartesian"); m_fft = 0; break;
		}
	printf ("\n");

	Attribute ("cgconv", &m_cgparam.cgconv);
	Attribute ("cgiter", &m_cgparam.cgiter);
	printf ("  Maximum %i NLCG iterations or convergence to m_cgconv", m_cgparam.cgiter, m_cgparam.cgconv);	

	

	m_initialised = true;
	printf ("... done.\n\n");

	return RRSModule::OK;

}


RRSModule::error_code
CompressedSensing::Process () {

	printf ("Processing CompressedSensing ...\n");
	ticks csstart = getticks();

	Matrix<cplx>&   data  = GetCplx("data");
	Matrix<double>& pdf   = GetReal("pdf");

	Matrix<cplx>&   im_dc = AddCplx ("im_dc", new Matrix<cplx> (data.Dim()));

	ticks tic = getticks();

	printf ("  Fourier transforming ... "); fflush(stdout);
	
	for (int i = 0; i < data.Size(); i++)
		data[i] /= pdf[i];
	
	data = FFT::Backward(data);

	printf ("done. (%.4f s)\n", elapsed(getticks(), tic) / Toolbox::Instance()->ClockRate());

	tic = getticks();

	cplx ma = data.Maxabs();
	data  /= ma;

	printf ("  Wavelet transforming ... "); fflush(stdout);

	im_dc = DWT::Forward (data);

	printf ("done. (%.4f s)\n", elapsed(getticks(), tic) / Toolbox::Instance()->ClockRate());

	tic = getticks();

	printf ("  Running CG ... "); fflush(stdout);

	for (int i = 0; i < m_csiter; i++) {
		
		NLCG (im_dc, data, m_cgparam);
		
	}
	
	printf ("done. (%.4f s)\n", elapsed(getticks(), tic) / Toolbox::Instance()->ClockRate());

	printf ("... done. WTime: %.4f seconds.\n\n", elapsed(getticks(), csstart) / Toolbox::Instance()->ClockRate());

	return OK;

}





// the class factories
extern "C" DLLEXPORT ReconStrategy* create  ()                 {
    return new CompressedSensing;
}


extern "C" DLLEXPORT void           destroy (ReconStrategy* p) {
    delete p;
}

