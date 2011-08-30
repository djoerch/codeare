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

#include "DumpToFile.hpp"

using namespace RRStrategy;

error_code 
DumpToFile::Process () {

	std::stringstream fname;
	const char* uid = Attribute ("UID");

	printf ("Dumping ...\n");

	if (uid == 0  ||  uid == "")
		uid = "unspecified";

	fname << uid << "_config.xml";
	DumpConfig (fname.str().c_str());

	fname.str("");

	fname << uid << "_data.mat";

	MATFile* mf = matOpen (fname.str().c_str(), "w");

	if (mf == NULL) {
		printf ("Error creating file %s\n", fname.str().c_str());
		return RRSModule::FILE_ACCESS_FAILED;
	}

	map<string,Matrix<cplx>*>::iterator cit = m_cplx.begin();
	for (cit = m_cplx.begin() ; cit != m_cplx.end(); cit++) {
		cout << "Dumping " << cit->first.c_str() << endl;
		cit->second->MXDump(mf, cit->first.c_str());
	}
	
	map<string,Matrix<double>*>::iterator rit = m_real.begin();
	for (rit = m_real.begin(); rit != m_real.end(); rit++) {
		cout << "Dumping " <<  rit->first.c_str() << endl;
		rit->second->MXDump(mf, rit->first.c_str());
	}
	
	map<string,Matrix<short>*>::iterator pit = m_pixel.begin();
	for (pit = m_pixel.begin(); pit != m_pixel.end(); pit++) {
		cout << "Dumping " <<  pit->first.c_str() << endl;
		pit->second->MXDump(mf, pit->first.c_str());
	}
	
	if (matClose(mf) != 0) {
		printf ("Error closing file %s\n",fname.str().c_str());
		return RRSModule::FILE_ACCESS_FAILED;
	}

	printf ("... done\n");

	return RRSModule::OK;

}

// the class factories
extern "C" DLLEXPORT ReconStrategy* create  ()                 {
    return new DumpToFile;
}

extern "C" DLLEXPORT void           destroy (ReconStrategy* p) {
    delete p;
}

