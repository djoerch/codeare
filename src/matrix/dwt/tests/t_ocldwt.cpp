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
typedef double elem_type;


using namespace codeare::matrix::io;



std::vector <int>
vector_of_sizes         (TiXmlElement * const elem)
{
  std::vector <int> values;
  if (strcmp (elem -> Attribute ("range"), "yes") == 0)
  {
    const int start_value = atoi (elem -> Attribute ("start_value"));
    const int   end_value = atoi (elem -> Attribute ("end_value"));
    const int   increment = atoi (elem -> Attribute ("increment"));
    for (int i = start_value; i <= end_value; i += increment)
      values.push_back (i);
  }
  else
  {
    TiXmlNode * child = NULL;
    while (child = elem -> IterateChildren (child))
    {
      values.push_back (atoi (child -> ToElement () -> GetText ()));
    }
  }
  return values;
}


void
pair_vector_of_sizes    (TiXmlElement * const elem, std::vector <std::pair <int, int> > & sizes)
{
  TiXmlElement * x_elem = elem -> FirstChildElement ("size_x");
  TiXmlElement * y_elem = elem -> FirstChildElement ("size_y");
  std::vector <int> x_values = vector_of_sizes (x_elem);
  std::vector <int> y_values = vector_of_sizes (y_elem);
  
  if (strcmp (elem->Attribute("permute"), "yes")==0)
  {
    for (std::vector <int> :: const_iterator it_x = x_values.begin (); it_x != x_values.end (); ++it_x)
      for (std::vector <int> :: const_iterator it_y = y_values.begin (); it_y != y_values.end (); ++it_y)
        sizes.push_back (std::pair <int, int> (*it_x, *it_y));
  }
  else
  {
    if (x_values.size () != y_values.size ())
    {
      throw oclError (" ! Unequal number of local or global sizes ! ", " pair_vector_of_sizes ()");
    }
    for (int i = 0; i < x_values.size(); i++)
      sizes.push_back (std::pair <int, int> (x_values [i], y_values [i]));
  }
}



/**
 * @brief              Extract local and global sizes from config file.
 * @param conf         Configurable.
 * @param local_sizes  Vector of pairs of local sizes.
 * @param global_sizes Vector of pairs of global sizes.
 */
void
extract_sizes           (const Configurable & conf,
                         std::vector <std::pair <int, int> > & local_sizes,
                         std::vector <std::pair <int, int> > & global_sizes)
{
  
  // get "local" node
  TiXmlElement * local_elem = conf.GetElement ("/config/gpu/local");
  pair_vector_of_sizes (local_elem, local_sizes);

  // get "global" node
  TiXmlElement * global_elem = conf.GetElement ("/config/gpu/global");
  pair_vector_of_sizes (global_elem, global_sizes);
  
  if (strcmp (conf.GetElement ("/config/gpu") -> Attribute ("permute"), "yes") == 0)
  {
    std::vector <std::pair <int, int> > tmp_local_sizes (local_sizes);
    std::vector <std::pair <int, int> > tmp_global_sizes (global_sizes);
    local_sizes.clear ();
    global_sizes.clear ();
    for (std::vector <std::pair <int, int> > :: const_iterator it_local = tmp_local_sizes.begin (); it_local != tmp_local_sizes.end (); ++it_local)
      for (std::vector <std::pair <int, int> > :: const_iterator it_global = tmp_global_sizes.begin (); it_global != tmp_global_sizes.end (); ++it_global)
      {
        local_sizes.push_back (*it_local);
        global_sizes.push_back (*it_global);
      }
  }
  
  if (local_sizes.size () != global_sizes.size ())
  {
    throw oclError (" ! Number of local and global sizes must be equal ! ", " extract_sizes ()");
  }
  
}



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
    const char * of_name_base =       conf.GetElement ("/config/ofname")->Attribute ("base");
    const char * name_param   =       conf.GetElement ("/config/ofname")->Attribute ("param");
    const char * range        =       conf.GetElement ("/config")->Attribute ("range");

    std::vector <std::pair <int, int> > local_sizes;
    std::vector <std::pair <int, int> > global_sizes;
    
    extract_sizes (conf, local_sizes, global_sizes);
    
    // open measurement output file
    ss.clear (), ss.str (std::string ());
    ss << base << of_name_base << "_wl_fam_" << wl_fam << "_wl_mem_" << wl_mem << "_wl_scale_" << wl_scale;
    if (strcmp (name_param, "global") == 0)
      ss << "_global_" << global_sizes [0].first << "_" << global_sizes [0].second << ".txt";
    else if (strcmp (name_param, "local") == 0)
      ss << "_local_" << local_sizes [0].first << "_" << local_sizes [0].second << ".txt";
    std::fstream fs;
    fs.open (ss.str ().c_str (), std::fstream::out);

    // adjust start value of loop index
    int init_num_threads = 1;
    if (!strcmp (range, "no"))
        init_num_threads = num_threads;

    oclConnection::Instance();

    // headline of table
    const int indent = 18;
    fs << "## OpenCL DWT ##" << std::endl;
    fs << setw (indent)   << "## Local size  --" << std::flush;
    fs << setw (indent)   << "  global_size  --" << std::flush <<
          setw (indent-2) << "  time exec  --" << std::flush <<
          setw (indent-2) << "  time mem  --" << std::flush <<
          setw (indent)   << "  bandwidth --" << std::flush <<
          setw (indent-2) << "  g2l --" << std::flush <<
          setw (indent-2) << "  l2g --" << std::flush <<
          setw (indent-2) << "  flops" << std::endl;

    Matrix <elem_type> mat_out_dwt (mat_in.Dim ());
    Matrix <elem_type> mat_out_dwt_recon (mat_in.Dim ());

    double time_ref = 0;
    std::vector <PerformanceInformation> vec_pi;
    
    for (int i = 0; i < local_sizes.size (); ++i)
    {
    
        // do something
        oclDWT <elem_type> dwt (mat_in.Dim (0), mat_in.Dim (1), mat_in.Dim (2), wl_fam, wl_mem, wl_scale, local_sizes [i].first, local_sizes [i].second, global_sizes [i].first, global_sizes [i].second);
        
        try
        {
          vec_pi = dwt.Trafo (mat_in, mat_out_dwt);

          std::vector <PerformanceInformation> tmp_vec = dwt.Adjoint (mat_out_dwt, mat_out_dwt_recon);
          
          vec_pi.insert (vec_pi.end(), tmp_vec.begin (), tmp_vec.end ());

        } catch (oclError & err)
        {
          std::cerr << " ! CAUTION: skipped (" << local_sizes [i].first << ", "
                  << local_sizes [i].second << ", " << global_sizes [i].first << ", "
                  << global_sizes [i].second << std::endl;
          std::cerr << err << std::endl;
          continue;
        }
          
//        mat_out_dwt_recon = mat_in;
        
          std::cout << std::endl;
          for (std::vector <PerformanceInformation> :: const_iterator it = vec_pi.begin ();
                  it != vec_pi.end (); ++it)
          {
            std::cout << *it << std::endl;
          }

        // make sure input data is correct for iteration ;) !!!
//        mat_out_dwt_recon = mat_in;


        PerformanceInformation pi = vec_pi [3];
//        for (int i = 0; i < iterations; i++)
//        {
//            pi += dwt.Trafo (mat_out_dwt_recon, mat_out_dwt) [3];
////            dwt.Adjoint (mat_out_dwt, mat_out_dwt_recon);
//        }

          std::cout << " -------------- " << std::endl;
          std::cout << pi << std::endl;
          std::cout << " -------------- " << std::endl;


        fs << setw (indent-5) << pi.lc.local_x << std::flush <<
              setw (indent) << pi.lc.global_x << std::flush << 
              setw (indent) << pi.time_exec << std::flush << 
              setw (indent) << (pi.time_mem_up + pi.time_mem_down) << std::flush <<
              setw (indent) << pi.parameter << std::flush <<
              setw (indent) << vec_pi [0].parameter << std::flush <<
              setw (indent) << vec_pi [1].parameter << std::flush <<
              setw (indent) << vec_pi [2].parameter << std::endl;

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
