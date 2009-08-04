#ifndef LIBTENSOR_SO_COPY_H
#define LIBTENSOR_SO_COPY_H

#include "defs.h"
#include "exception.h"
#include "core/symmetry_i.h"
#include "symmetry_target.h"
#include "default_symmetry.h"

namespace libtensor {

template<size_t N, typename T>
class so_copy {
private:
	class symtgt :
		public symmetry_const_target< N, T, default_symmetry<N, T> > {
	private:
		symmetry_i <N, T> *m_cp;
	public:
		symtgt() : m_cp(NULL) { }
		~symtgt() { delete m_cp; }
		symmetry_i<N, T> *get_symmetry() { return m_cp; }
		virtual void accept(const default_symmetry<N, T> &sym)
			throw(exception);
	};

private:
	const symmetry_i<N, T> &m_src; //!< Source symmetry object
	symtgt m_tgt; //!< Target for symmetry dispatch

public:
	so_copy(const symmetry_i<N, T> &src);
	virtual ~so_copy() { };
	symmetry_i<N, T> &get_symmetry() throw(exception);
};


template<size_t N, typename T>
so_copy<N, T>::so_copy(const symmetry_i<N, T> &src) : m_src(src) {

	m_src.dispatch(m_tgt);
}


template<size_t N, typename T>
symmetry_i<N, T> &so_copy<N, T>::get_symmetry() throw(exception) {

	symmetry_i<N, T> *psym = m_tgt.get_symmetry();
	if(psym == NULL) {
		throw_exc("so_copy<N, T>", "get_symmetry()", "NULL pointer");
	}
	return *psym;
}


template<size_t N, typename T>
void so_copy<N, T>::symtgt::accept(const default_symmetry<N, T> &sym) {

	if(m_cp) delete m_cp;
	m_cp = new default_symmetry<N, T>(sym);
}

} // namespace libtensor

#endif // LIBTENSOR_SO_COPY_H
