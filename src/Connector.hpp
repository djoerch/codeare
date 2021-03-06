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

#ifndef __CONNECTOR_HPP__
#define __CONNECTOR_HPP__

#include "Matrix.hpp"

namespace RRClient {

/**
 * @brief Connector skeleton
 *        Abstraction layer for local or remote access to reconstruction schemes
 */
template <class T>
class Connector {

public:
	

	/**
	 * @brief  Default constructor 
	 * @see    Connector (const connection_type)
	 */
	Connector () : m_conn() {

	}
	
	
	/**
	 * @brief       Construct with service name and debug level.
	 *
	 * @param name  Service name
	 * @param debug Trace level
	 */
	Connector       (const char* name, const char* debug) {

		m_name  = name;
		m_debug = debug;

		Connect ();

	}
	
	
	/**
	 * @brief      Close connection
	 */
	virtual 
	~Connector () {
		delete m_conn;
	}


	/**
	 * @brief      Connect
	 */
	inline void 
	Connect() {

		m_conn  = new T (m_name.c_str(), m_debug.c_str());

	}


	/**
	 * @brief           Request data procession on remote service
	 *
	 *                  @see RRStrategy::ReconStrategy::Process()
	 *
	 * @param  name     Recon method
	 * @return          Error code
	 */ 
	virtual inline error_code              
	Process             (const char* name) {
		return (error_code) m_conn->Process(name);
	}
	
	
	/**
	 * @brief           Prepare backend
	 *
	 *                  @see RRStrategy::ReconStrategy::Prepare()
	 *
	 * @param  name     Recon method
	 * @return          Error code
	 */ 
	virtual inline error_code              
	Prepare             (const char* name) {
		return (error_code) m_conn->Prepare(name);
	}
	
	
	/**
	 * @brief           Initialise remote service
	 *
	 *                  @see RRStrategy::ReconStrategy::Init()
	 *
	 * @param  name     Recon method
	 * @return          Error code
	 */ 
	virtual inline error_code              
	Init                (const char* name) {
		return (error_code) m_conn->Init(name);
	}
	
	
	/**
	 * @brief           Finalise remote service
	 *
	 *                  @see RRStrategy::ReconStrategy::Finalise()
	 *
	 * @param  name     Recon method
	 * @return          Error error
	 */ 
	virtual inline error_code              
	Finalise            (const char* name) {
		return (error_code) m_conn->Finalise(name);
	}
	
	
	/**
	 * @brief           Transmit measurement data to remote service
	 *
	 *                  @see Database::SetMatrix
	 *
	 * @param  name     Name
	 * @param  m        Matrix
	 */
	template <class S> inline void 
	SetMatrix           (const std::string& name, Matrix<S>& m) const {
		m_conn->SetMatrix (name, m);
	}
	
	
	/**
	 * @brief           Retrieve manipulated data from remote service
	 *
	 *                  @see Database::GetMatrix
	 *
	 * @param  name     Name
	 * @param  m        Receive storage
	 */
	template <class S> inline void 
	GetMatrix           (const std::string& name, Matrix<S>& m) const {
		m_conn->GetMatrix (name, m);
	}
		
		
	/**
	 * @brief          Read configuration 
	 *
	 *                 @see Configurable::ReadConfig(const char* fname)
	 *                 @see Configurable::ReadConfig(FILE* file)
	 *
	 * @param config   Name of input file or file access pointer
	 */
	template <class S> inline void 
	ReadConfig        (S config) {
		m_conn->ReadConfig (config);
	}
	

	/**
	 * @brief           Set a string type attribute
	 *
	 *                  @see Configurable::SetAttribute
	 *
	 * @param  name     Attribute name 
	 * @param  value    Attribute value
	 */
	template <class S> inline void
	SetAttribute        (const char* name, S value) {
		m_conn->SetAttribute (name, value);
	}

	
	/**
	 * @brief           Set a string type attribute
	 *
	 *                  @see Configurable::Attribute
	 *
	 * @param  name     Attribute name 
	 * @param  value    Attribute value
	 */
	template <class S> inline int
	Attribute        (const char* name, S* value) {
		return m_conn->Attribute (name, value);
	}

	
	/**
	 * @brief           Get a string type attribute
	 *
	 *                  @see Configurable::Attribute
	 *
	 * @param  name     Attribute name 
	 * @return          Attribute value
	 */
	inline const char*
	Attribute          (const char* name) {
		return m_conn->Attribute (name);
	}

	
	/**
	 * @brief           Get a text node of an element
	 *
	 *                  @see Configurable::GetText
	 *
	 * @param  path     X-Path
	 * @return          Text
	 */
	inline const char*
	GetText            (const char* path) {
		return m_conn->GetText (path);
	}

	
	/**
	 * @brief           Get a text node of an element
	 *
	 *                  @see Configurable::GetText
	 *
	 * @param  path     X-Path
	 * @return          Text
	 */
	inline TiXmlElement*
	GetElement          (const char* path) {
		return m_conn->GetElement (path);
	}

	

private:
	
	
	T*          m_conn; /**< Actual connection */
	std::string m_name;
	std::string m_debug;
	
};
	

}
#endif
