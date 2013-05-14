/**************
 ** includes **
 **************/
# include "matrix/ocl/oclmatrix/oclMatrix.hpp"


/**********************
 ** type definitions **
 **********************/
typedef double elem_type;


/*******************
 ** test function **
 *******************/
template <class T> bool
ocldwttest (RRClient::Connector<T>* rc)
{

	// Intro
	std::cout << std::endl;
	std::cout << " * Running test: \"ocl_dwt\"!";
	std::cout << std::endl;


	// create oclMatrix from input file
	Matrix<elem_type> mat_in;
	if (!Read (mat_in, rc->GetElement ("/config/data/in"), base))
	{
		std::cerr << " *!* Error while reading input matrix!" << std::endl << std::endl;
		return false;
	}

	std::string ofname (base + std::string (rc->GetElement ("/config/data/out") -> Attribute ("fname")));
	std::string odname (rc->GetElement ("/config/data/out") -> Attribute ("dname"));


	// do something


	// output oclMatrix to output file

#ifdef HAVE_MAT_H
	MATFile* mf = matOpen (ofname.c_str(), "w");

	if (mf == NULL) {
		printf ("Error creating file %s\n", ofname.c_str());
		return false;
	}

	MXDump     (mat_in, mf, odname);

	if (matClose(mf) != 0) {
		printf ("Error closing file %s\n", ofname.c_str());
		return false;
	}
#endif


	// Outro
	std::cout << std::endl;
	std::cout << " * Finished test: \"ocl_dwt\"!";
	std::cout << std::endl;

}
