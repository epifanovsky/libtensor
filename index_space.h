#ifndef LIBTENSOR_INDEX_SPACE_H
#define LIBTENSOR_INDEX_SPACE_H

#include "defs.h"
#include "exception.h"
#include "index.h"

namespace libtensor {

/**	\brief Defines an %index space with a given symmetry
	\param N Space rank
	\param Sym Space symmetry

	<b>Rank and dimension</b>

	If an %index space has rank <i>n</i> and dimension <i>d</i>, each
	element of the space (each %index) contains <i>n</i> integers from
	the range [0,<i>d</i>-1].
**/
template<size_t N, typename Sym>
class index_space {
public:
	typedef index<N> index_t; //!< Index type

private:
	size_t m_d; //!< Space dimension

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Constructs an %index space with specified dimension
		\param d Space dimension
	**/
	index_space(size_t d);

	/**	\brief Copy constructor
		\param is Another %index space
	**/
	index_space(const index_space<N,Sym> &is);

	//@}
};

template<size_t N, typename Sym>
inline index_space<N,Sym>::index_space(size_t d) : m_d(d) {
}

template<size_t N, typename Sym>
inline index_space<N,Sym>::index_space(const index_space<N,Sym> &is) :
	m_d(is.m_d) {
}

} // namespace libtensor

#endif // LIBTENSOR_INDEX_SPACE_H

