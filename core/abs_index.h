#ifndef LIBTENSOR_ABS_INDEX_H
#define LIBTENSOR_ABS_INDEX_H

#include "defs.h"
#include "exception.h"
#include "index.h"
#include "dimensions.h"

namespace libtensor {


/**	\brief Absolute %index within %dimensions
	\tparam N Tensor order.

	\ingroup libtensor_core
 **/
template<size_t N>
class abs_index {
public:
	static const char *k_clazz; //!< Class name

private:
	dimensions<N> m_dims; //!< Dimensions
	index<N> m_idx; //!< Regular %index
	size_t m_abs_idx; //!< Absolute %index

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the first %index within %dimensions
	 **/
	abs_index(const dimensions<N> &dims);

	/**	\brief Initializes an %index within %dimensions
	 **/
	abs_index(const index<N> &idx, const dimensions<N> &dims);

	/**	\brief Initializes an %index within %dimensions
	 **/
	abs_index(size_t abs_idx, const dimensions<N> &dims);

	/**	\brief Copy constructor
	 **/
	abs_index(const abs_index<N> &abs_idx);

	//@}


	//!	\name Manipulations
	//@{

	/**	\brief Returns the %dimensions
	 **/
	const dimensions<N> &get_dims() const;

	/**	\brief Returns the %index
	 **/
	const index<N> &get_index() const;

	/**	\brief Returns the absolute %index
	 **/
	size_t get_abs_index() const;

	/**	\brief Increments the current %index, returns true if success
	 **/
	bool inc();

	/**	\brief Returns whether the current value is the last %index
			within the %dimensions
	 **/
	bool is_last() const;

	/**	\brief Increments the current %index
	 **/
	abs_index<N> &operator++();

	//@}
};


template<size_t N>
const char *abs_index<N>::k_clazz = "abs_index<N>";


template<size_t N>
abs_index<N>::abs_index(const dimensions<N> &dims) :
	m_dims(dims), m_abs_idx(0) {

}


template<size_t N>
abs_index<N>::abs_index(const index<N> &idx, const dimensions<N> &dims) :
	m_dims(dims), m_idx(idx) {

	static const char *method = "abs_index(const index<N>&, "
		"const dimensions<N>&)";

	m_abs_idx = 0;
	for(register size_t i = 0; i < N; i++) {
		if(m_idx[i] >= m_dims[i]) {
			throw out_of_bounds(g_ns, k_clazz, method, __FILE__,
				__LINE__, "Index out of range.");
		}
		m_abs_idx += m_dims.get_increment(i)*m_idx[i];
	}
}


template<size_t N>
abs_index<N>::abs_index(size_t abs_idx, const dimensions<N> &dims) :
	m_dims(dims), m_abs_idx(abs_idx) {

	static const char *method = "abs_index(size_t, const dimensions<N>&)";

	if(m_abs_idx >= m_dims.get_size()) {
		throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Index out of range.");
	}

	size_t a = m_abs_idx;
	register size_t imax = N - 1;
	for(register size_t i = 0; i < imax; i++) {
		m_idx[i] = a / m_dims.get_increment(i);
		a %= m_dims.get_increment(i);
	}
	m_idx[N - 1] = a;
}


template<size_t N>
abs_index<N>::abs_index(const abs_index<N> &abs_idx) :
	m_dims(abs_idx.m_dims), m_idx(abs_idx.m_idx),
	m_abs_idx(abs_idx.m_abs_idx) {

}


template<size_t N>
inline const dimensions<N> &abs_index<N>::get_dims() const {

	return m_dims;
}


template<size_t N>
inline const index<N> &abs_index<N>::get_index() const {

	return m_idx;
}


template<size_t N>
inline size_t abs_index<N>::get_abs_index() const {

	return m_abs_idx;
}


template<size_t N>
bool abs_index<N>::inc() {

	if(m_abs_idx + 1 >= m_dims.get_size()) return false;

	size_t n = N - 1;
	bool done = false, ok = false;
	do {
		if(m_idx[n] < m_dims[n] - 1) {
			m_idx[n]++;
			for(register size_t i = n + 1; i < N; i++) m_idx[i]=0;
			done = true; ok = true;
		} else {
			if(n == 0) done = true;
			else n--;
		}
	} while(!done);
	if(ok) m_abs_idx++;
	return ok;
}


template<size_t N>
inline bool abs_index<N>::is_last() const {

	return m_abs_idx + 1 >= m_dims.get_size();
}


template<size_t N>
inline abs_index<N> &abs_index<N>::operator++() {

	inc();
	return *this;
}


} // namespace libtensor

#endif // LIBTENSOR_ABS_INDEX_H
