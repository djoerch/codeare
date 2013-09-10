/**************
 ** includes **
 **************/
# include "Matrix.hpp"
# include "io/IOContext.hpp"
# include "dwt/oclDWT.hpp"
# include "Configurable.hpp"
# include <sstream>


/**********************
 ** type definitions **
 **********************/
typedef float elem_type;


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

    const int start_local_size  = atoi (conf.GetElement ("/config/gpu/local")->Attribute ("start_value"));
    const int end_local_size  = atoi (conf.GetElement ("/config/gpu/local")->Attribute ("end_value"));
    const int start_global_size  = atoi (conf.GetElement ("/config/gpu/global")->Attribute ("start_value"));
    const int end_num_groups  = atoi (conf.GetElement ("/config/gpu/global")->Attribute ("end_value"));
    
    // open measurement output file
    ss.clear (), ss.str (std::string ());
    ss << base << of_name_base << "_wl_fam_" << wl_fam << "_wl_mem_" << wl_mem << "_wl_scale_" << wl_scale << "_gs_" << start_global_size << ".txt";
    std::fstream fs;
    fs.open (ss.str ().c_str (), std::fstream::out);

    // adjust start value of loop index
    int init_num_threads = 1;
    if (!strcmp (range, "no"))
        init_num_threads = num_threads;

    oclConnection::Instance();

    // headline of table
    fs << "## OpenCL DWT ##" << std::endl;
    fs << "## Local size  --  global_size  --  time exec  --  time mem  --  bandwidth" << std::endl;

    Matrix <elem_type> mat_out_dwt (mat_in.Dim ());
    Matrix <elem_type> mat_out_dwt_recon (mat_in.Dim ());

    double time_ref = 0;
    std::vector <PerformanceInformation> vec_pi;
    

    for (int global_size = start_global_size; global_size <= end_num_groups; global_size *= 2)
    {
      for (int local_size = start_local_size; local_size <= end_local_size; local_size *= 2)
      {

        // do something
        oclDWT <elem_type> dwt (mat_in.Dim (0), mat_in.Dim (1), mat_in.Dim (2), wl_fam, wl_mem, wl_scale, local_size, global_size);
        vec_pi = dwt.Trafo (mat_in, mat_out_dwt);

        dwt.Adjoint (mat_out_dwt, mat_out_dwt_recon);

//          std::cout << std::endl;
//          for (std::vector <PerformanceInformation> :: const_iterator it = vec_pi.begin ();
//                  it != vec_pi.end (); ++it)
//          {
//            std::cout << *it << std::endl;
//          }

        // make sure input data is correct for iteration ;) !!!
//        mat_out_dwt_recon = mat_in;


        PerformanceInformation pi = vec_pi [3];
        for (int i = 0; i < iterations; i++)
        {
            pi += dwt.Trafo (mat_out_dwt_recon, mat_out_dwt) [3];
//            dwt.Adjoint (mat_out_dwt, mat_out_dwt_recon);
        }

          std::cout << " -------------- " << std::endl;
          std::cout << pi << std::endl;
          std::cout << " -------------- " << std::endl;


        fs << "\t" << pi.lc.local_x << "\t\t" << pi.lc.global_x << "\t\t" << pi.time_exec << "\t\t" << (pi.time_mem_up + pi.time_mem_down) << "\t\t" << pi.parameter << std::endl;

      }
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
