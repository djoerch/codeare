/**************
 ** includes **
 **************/
# include "Matrix.hpp"
# include "io/IOContext.hpp"
# include "oclDWT.hpp"
# include "Configurable.hpp"
# include <sstream>


/**********************
 ** type definitions **
 **********************/
typedef double elem_type;


using namespace codeare::matrix::io;



/*******************
 ** test function **
 *******************/
int
main            (int argc, char ** args)
{

    if (argc != 3)
    {
        std::cerr << " usage: t_ocldwt <base_dir> <config_file> " << std::endl;
        exit (-1);
    }

    char * base = args [1];

    std::stringstream ss;
    ss << base << args [2];

    Configurable conf;
    conf.ReadConfig (ss.str ().c_str ());

    // Intro
    std::cout << std::endl;
    std::cout << " * Running test: \"t_ocldwt\"!";
    std::cout << std::endl;
    std::cout << std::endl;


    // create oclMatrix from input file
    Matrix <elem_type> mat_in;
    IOContext ioc (conf.GetElement ("/config/data/in"), base, READ);
    mat_in = ioc.Read <elem_type> (conf.GetElement ("/config/data/in/signal"));

    // read DWT params
    wlfamily  wl_fam   = (wlfamily) atoi (conf.GetElement ("/config/dwt")->Attribute ("wl_fam"));
    const int wl_mem   =            atoi (conf.GetElement ("/config/dwt")->Attribute ("wl_mem"));
    const int wl_scale =            atoi (conf.GetElement ("/config/dwt")->Attribute ("wl_scale"));

    // more params
    const int    iterations   = atoi (conf.GetElement ("/config")->Attribute ("iterations"));
    const int    num_threads  = 1; //atoi (conf.GetElement ("/config")->Attribute ("num_threads"));
    const char * of_name_base =       conf.GetElement ("/config")->Attribute ("ofname");
    const char * range        =       conf.GetElement ("/config")->Attribute ("range");

    // open measurement output file
    ss.clear (), ss.str (std::string ());
    ss << base << of_name_base << "_wl_fam_" << wl_fam << "_wl_mem_" << wl_mem << "_wl_scale_" << wl_scale << ".txt";
    std::fstream fs;
    fs.open (ss.str ().c_str (), std::fstream::out);

    // adjust start value of loop index
    int init_num_threads = 1;
    if (!strcmp (range, "no"))
        init_num_threads = num_threads;

    // headline of table
    fs << "## OpenCL DWT ##" << std::endl;
    fs << "## No. of threads  --  per transform:  --  single forward:  --  single backwards:  --  S(p)" << std::endl;

    Matrix <elem_type> mat_out_dwt (mat_in.Dim ());
    Matrix <elem_type> mat_out_dwt_recon (mat_in.Dim ());

    double time_ref = 0;

    // loop over number of threads
    for (int threads = init_num_threads; threads <= num_threads; threads +=1) //*= 2)
    {

        // do something
        oclDWT <elem_type> dwt (mat_in.Dim (0), mat_in.Dim (1), mat_in.Dim (2), wl_fam, wl_mem, wl_scale);
        double s_time_f = omp_get_wtime ();
        dwt.Trafo (mat_in, mat_out_dwt);
        s_time_f = omp_get_wtime () - s_time_f;
        double s_time_b = omp_get_wtime ();
        dwt.Adjoint (mat_out_dwt, mat_out_dwt_recon);
        s_time_b = omp_get_wtime () - s_time_b;

        // make sure input data is correct for iteration ;) !!!
//      mat_out_dwt_recon = mat_in;

        double time = omp_get_wtime ();
        for (int i = 0; i < iterations; i++)
        {
            dwt.Trafo (mat_out_dwt_recon, mat_out_dwt);
            dwt.Adjoint (mat_out_dwt, mat_out_dwt_recon);
        }
        time = omp_get_wtime () - time;

        if (threads == 1)
            time_ref = time;

        fs << "\t" << threads << "\t\t" << time/iterations/2 << "\t\t" << s_time_f << "\t\t" << s_time_b << "\t\t" << time_ref/time << std::endl;

    }

    // output oclMatrix to output file
    IOContext ioc2 (conf.GetElement ("/config/data/out"), base, WRITE);
    ioc2.Write (mat_out_dwt,       conf.GetElement ("/config/data/out/res-dwt"));
    ioc2.Write (mat_out_dwt_recon, conf.GetElement ("/config/data/out/res-dwt-recon"));

    fs.close ();

    // Outro
    std::cout << std::endl;
    std::cout << " * Finished test: \"t_ocldwt\"!";
    std::cout << std::endl;
    std::cout << std::endl;

}
