/**************
 ** includes **
 **************/
# include "Matrix.hpp"
# include "io/IOContext.hpp"
# include "dwt/oclDWT.hpp"
# include "Configurable.hpp"
# include <sstream>
# include "boost/tuple/tuple.hpp"
# include "boost/tuple/tuple_comparison.hpp"


/**********************
 ** type definitions **
 **********************/
typedef float elem_type;


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
      if (child -> Type() == TiXmlNode::TINYXML_COMMENT)
        continue;
      values.push_back (atoi (child -> ToElement () -> GetText ()));
    }
  }
  return values;
}


void
tuple_vector_of_sizes    (TiXmlElement * const elem, std::vector <boost::tuple <int, int, int> > & sizes)
{
  TiXmlElement * x_elem = elem -> FirstChildElement ("size_x");
  TiXmlElement * y_elem = elem -> FirstChildElement ("size_y");
  TiXmlElement * z_elem = elem -> FirstChildElement ("size_z");
  std::vector <int> x_values = vector_of_sizes (x_elem);
  std::vector <int> y_values = vector_of_sizes (y_elem);
  std::vector <int> z_values = vector_of_sizes (z_elem);
  if (strcmp (elem->Attribute("permute"), "yes")==0)
  {
    for (std::vector <int> :: const_iterator it_x = x_values.begin (); it_x != x_values.end (); ++it_x)
      for (std::vector <int> :: const_iterator it_y = y_values.begin (); it_y != y_values.end (); ++it_y)
        for (std::vector <int> :: const_iterator it_z = z_values.begin (); it_z != z_values.end (); ++it_z)
          sizes.push_back (boost::make_tuple (*it_x, *it_y, *it_z));
  }
  else
  {
    if (x_values.size () != y_values.size () || y_values.size () != z_values.size ())
    {
      throw oclError (" ! Unequal number of local or global sizes ! ", " tuple_vector_of_sizes ()");
    }
    for (int i = 0; i < x_values.size(); i++)
      sizes.push_back (boost::make_tuple (x_values [i], y_values [i], z_values [i]));
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
                         std::vector <boost::tuple <int, int, int> > & local_sizes,
                         std::vector <boost::tuple <int, int, int> > & global_sizes,
                         const std::string & node_name)
{
  
  // get "local" node
  TiXmlElement * local_elem = conf.GetElement ((node_name + std::string ("/local")).c_str ());
  tuple_vector_of_sizes (local_elem, local_sizes);
  
  // get "global" node
  TiXmlElement * global_elem = conf.GetElement ((node_name + std::string ("/global")).c_str ());
  tuple_vector_of_sizes (global_elem, global_sizes);
  
  if (strcmp (conf.GetElement ("/config/gpu") -> Attribute ("permute"), "yes") == 0)
  {
    std::vector <boost::tuple <int, int, int> > tmp_local_sizes (local_sizes);
    std::vector <boost::tuple <int, int, int> > tmp_global_sizes (global_sizes);
    local_sizes.clear ();
    global_sizes.clear ();
    for (std::vector <boost::tuple <int, int, int> > :: const_iterator it_local = tmp_local_sizes.begin (); it_local != tmp_local_sizes.end (); ++it_local)
      for (std::vector <boost::tuple <int, int, int> > :: const_iterator it_global = tmp_global_sizes.begin (); it_global != tmp_global_sizes.end (); ++it_global)
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


void
print_table_header (std::fstream & fs, const int iterations, const char * name_param, const int indent)
{
  
    fs << "## OpenCL DWT ##" << std::endl;
    fs << "## iterations: " << iterations << std::endl;
    fs << "## fixed " << name_param << " configuration" << std::endl;
  
    // headline of table
    fs << " ##";
    if (strcmp (name_param, "local") == 0)
    {
      fs               << " global_size_x  --" << std::flush <<
         setw (indent) << " global_size_y  --" << std::flush <<
         setw (indent) << " global_size_z  --" << std::flush;
    }
    else if (strcmp (name_param, "global") == 0)
    {
      fs << setw (indent)   << " local_size_x  --" << std::flush <<
            setw (indent)   << " local_size_y  --" << std::flush <<
            setw (indent)   << " local_size_z  --" << std::flush;
    }
    fs << setw (indent-2) << "  time exec (f)  --" << std::flush <<
          setw (indent-2) << "  time exec (b)  --" << std::flush <<
          setw (indent-2) << "  time mem  --" << std::flush <<
          setw (indent)   << "  bandwidth (f) --" << std::flush <<
          setw (indent)   << "  bandwidth (b) --" << std::flush <<
          setw (indent-2) << "  g2l (f) --" << std::flush <<
          setw (indent-2) << "  l2g (f) --" << std::flush <<
          setw (indent-2) << "  flops (f) " << std::endl;

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
    const char * of_path      =       conf.GetElement ("/config/ofname")->Attribute ("path");
    const char * of_name_base =       conf.GetElement ("/config/ofname")->Attribute ("base");
    const char * name_param   =       conf.GetElement ("/config/ofname")->Attribute ("param");
    
    std::vector <boost::tuple <int, int, int> > local_sizes;
    std::vector <boost::tuple <int, int, int> > global_sizes;
    extract_sizes (conf, local_sizes, global_sizes, std::string ("/config/gpu/dwt2"));
    
    std::vector <boost::tuple <int, int, int> > dwt_local;
    std::vector <boost::tuple <int, int, int> > dwt_global;
    extract_sizes (conf, dwt_local, dwt_global, std::string ("/config/gpu/dwt3"));
    
    const int chunk_size = atoi (conf.GetElement ("/config/gpu/dwt2")->Attribute ("chunk_size"));
    
    // open measurement output file
    ss.clear (), ss.str (std::string ());
    ss << base << of_path << "ocldwt2_";
    ss << of_name_base << "_wl_fam_" << wl_fam << "_wl_mem_" << wl_mem << "_wl_scale_" << wl_scale;
    ss << "_cs_" << chunk_size;
    if (strcmp (name_param, "global") == 0)
      ss << "_global_" << global_sizes [0].get <0> () << "_" << global_sizes [0].get <1> () << ".txt";
    else if (strcmp (name_param, "local") == 0)
      ss << "_local_" << local_sizes [0].get <0> () << "_" << local_sizes [0].get <1> () << ".txt";
    std::fstream fs;
    fs.open (ss.str ().c_str (), std::fstream::out);
    
    oclConnection::Instance();

    const int indent = 20;
    print_table_header (fs, iterations, name_param, indent);

    Matrix <elem_type> mat_out_dwt (mat_in.Dim ());
    Matrix <elem_type> mat_out_dwt_recon (mat_in.Dim ());

    double time_ref = 0;
    
    for (int i = 0; i < local_sizes.size (); ++i)
    {

      // for formatted table output
      boost::tuple <int, int, int> sizes = (0==strcmp(name_param, "local") ? local_sizes [i] : global_sizes [i]);
      
      try
      {
        
        const LaunchInformation lc1 (local_sizes [i].get <0> (), local_sizes [i].get <1> (), local_sizes [i].get <2> (),
                                     global_sizes [i].get <0> (), global_sizes [i].get <1> (), global_sizes [i].get <2> ());
        const LaunchInformation lc2 (dwt_local [0].get <0> (), dwt_local [0].get <1> (), dwt_local [0].get <2> (),
                                     dwt_global [0].get <0> (), dwt_global [0].get <1> (), dwt_global [0].get <2> ());
                                     
        // do something
        oclDWT <elem_type> dwt (mat_in.Dim (0), mat_in.Dim (1), mat_in.Dim (2), wl_fam, wl_mem, wl_scale, lc1, lc2);
        
        std::vector <PerformanceInformation> vec_pi_forward = dwt.Trafo (mat_in, mat_out_dwt, chunk_size);
//        std::vector <PerformanceInformation> vec_pi_backwards = dwt.Adjoint (mat_out_dwt, mat_out_dwt_recon, chunk_size);
        
          
//        mat_out_dwt_recon = mat_in;
        
//          std::cout << std::endl;
//          for (std::vector <PerformanceInformation> :: const_iterator it = vec_pi_forward.begin ();
//                  it != vec_pi_forward.end (); ++it)
//          {
//            std::cout << *it << std::endl;
//          }

        // make sure input data is correct for iteration ;) !!!
//        mat_out_dwt_recon = mat_in;

        for (int j = 1; j < iterations; j++)
        {
            std::vector <PerformanceInformation> vec_tmp_1 = dwt.Trafo (mat_out_dwt_recon, mat_out_dwt);
            std::vector <PerformanceInformation> vec_tmp_2 = dwt.Adjoint (mat_out_dwt, mat_out_dwt_recon);
            for (int k = 0; k < vec_tmp_1.size(); k++)
              vec_pi_forward [k] += vec_tmp_1 [k];
//            for (int k = 0; k < vec_tmp_2.size(); k++)
//              vec_pi_backwards [k] += vec_tmp_2 [k];
        }
        
//          std::cout << std::endl;
//          for (std::vector <PerformanceInformation> :: const_iterator it = vec_pi_forward.begin ();
//                  it != vec_pi_forward.end (); ++it)
//          {
//            std::cout << *it << std::endl;
//          }

          PerformanceInformation pi = vec_pi_forward [0];
          std::cout << " -------------- " << std::endl;
          std::cout << pi << std::endl;
          std::cout << " -------------- " << std::endl;

          PerformanceInformation tmp_pi = vec_pi_forward [1];
          std::cout << " -------------- " << std::endl;
          std::cout << tmp_pi << std::endl;
          std::cout << " -------------- " << std::endl;

          tmp_pi = vec_pi_forward [2];
          std::cout << " -------------- " << std::endl;
          std::cout << tmp_pi << std::endl;
          std::cout << " -------------- " << std::endl;
          
//          PerformanceInformation pi2 = vec_pi_backwards [0];
//          std::cout << " -------------- " << std::endl;
//          std::cout << pi2 << std::endl;
//          std::cout << " -------------- " << std::endl;

          fs << setw (indent-5) << (strcmp (name_param, "local")?pi.lc.local_x:pi.lc.global_x) << std::flush <<
              setw (indent) << (strcmp (name_param, "local")?pi.lc.local_y:pi.lc.global_y) << std::flush <<
              setw (indent) << (strcmp (name_param, "local")?pi.lc.local_z:pi.lc.global_z) << std::flush <<
              setw (indent) << pi.time_exec << std::flush <<
//              setw (indent) << pi2.time_exec << std::flush <<
              setw (indent) << (pi.time_mem_up + pi.time_mem_down) << std::flush <<
              setw (indent) << pi.parameter << std::flush <<
//              setw (indent) << pi2.parameter << std::flush <<
              setw (indent) << vec_pi_forward [0].parameter << std::flush <<
              setw (indent) << vec_pi_forward [1].parameter << std::flush <<
              setw (indent) << vec_pi_forward [2].parameter << std::endl;
          if (i < local_sizes.size () - 1 && (0==strcmp (name_param, "local") ? local_sizes [i+1] : global_sizes [i+1]) != sizes)
            fs << std::endl;

                  
        } catch (oclError & err)
        {
          std::cerr << " ! CAUTION: skipped (" << local_sizes [i].get <0> () << ", "
                  << local_sizes [i].get <1> () << ", " << local_sizes [i].get <2> () << ", "
                  << global_sizes [i].get <0> () << ", " << global_sizes [i].get <1> () << ", "
                  << global_sizes [i].get <2> () << ")" << std::endl;
          std::cerr << err << std::endl;
          continue;
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
