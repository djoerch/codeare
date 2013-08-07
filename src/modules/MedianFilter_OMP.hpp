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

#ifndef __MEDIAN_FILTER_OMP_HPP__
#define __MEDIAN_FILTER_OMP_HPP__

#include "ReconStrategy.hpp"

namespace RRStrategy {

	/**
	 * @brief Median filter with OpenMP support
	 */
	class MedianFilter_OMP : public ReconStrategy {
		
		
	public:
		
		/**
		 * @brief Default constructor
		 */
		MedianFilter_OMP  () {};
		
		/**
		 * @brief Default destructor
		 */
		virtual 
		~MedianFilter_OMP () {};
		
		/**
		 * @brief Apply Median filter to image space
		 */
		virtual error_code
		Process ();
		
		/**
		 * @brief Do nothing 
		 */
		virtual error_code
		Init () ;
		
		/**
		 * @brief Do nothing 
		 */
		virtual error_code
		Finalise () {
			return OK;
		}

    private:

        unsigned short m_ww;
        unsigned short m_wh;
        std::string m_uname;
		
	};

}
#endif /* __MEDIAN_FILTER_OMP_HPP__ */
