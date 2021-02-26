#ifndef LIBUTIL_LOADED_COND_H
#define LIBUTIL_LOADED_COND_H

#include "cond.h"

namespace libutil {


/**	\brief Loaded conditional variable

	In addition to synchronizing of two threads, the loaded conditional
	variable facilitates the transfer of data (load). The signaling thread 
	puts the load before calling signal(), and the waiting thread reads
	the load afterward.

	The waiting/signaling interface is the same as in the usual conditional
	variable.

	\sa cond

	\ingroup libutil
 **/
template<typename LoadT>
class loaded_cond {
public:
	typedef LoadT load_t; //!< Load type

private:
	cond m_cond; //!< Conditional variable
	load_t m_load; //!< Load

public:
	/**	\brief Initializing constructor
		\param load Default load.
	 **/
	loaded_cond(const load_t &load) : m_load(load) { }

	/**	\brief Waits for the condition
	 **/
	void wait() {
		m_cond.wait();
	}

	/**	\brief Signals the condition
	 **/
	void signal() {
		m_cond.signal();
	}

	/**	\brief Returns the const reference to the load
	 **/
	const load_t &load() const {
		return m_load;
	}

	/**	\brief Returns the reference to the load
	 **/
	load_t &load() {
		return m_load;
	}

private:
	loaded_cond(const loaded_cond&);
	loaded_cond &operator=(const loaded_cond&);

};


} // namespace libutil

#endif // LIBUTIL_LOADED_COND_H

