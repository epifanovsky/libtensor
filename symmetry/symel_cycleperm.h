#ifndef LIBTENSOR_SYMEL_CYCLEPERM_H
#define LIBTENSOR_SYMEL_CYCLEPERM_H

#include "defs.h"
#include "exception.h"
#include "core/dimensions.h"
#include "symmetry_element_base.h"
#include "btod/transf_double.h"

namespace libtensor {

/**	\brief Symmetry element: cyclic permutation
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symel_cycleperm :
	public symmetry_element_base< N, T, symel_cycleperm<N, T> > {

public:
	static const char *k_clazz; //!< Class name

private:
	class equals_functor :
		public symmetry_element_target< N, T, symel_cycleperm<N, T> > {
	private:
		const symel_cycleperm<N, T> &m_elem;
		bool m_eq;
	public:
		equals_functor(const symel_cycleperm<N, T> &elem)
			: m_elem(elem), m_eq(false) { };
		virtual ~equals_functor() { };
		virtual void accept_default(
			const symmetry_element_i<N, T> &elem) throw(exception) {
			m_eq = false;
		}
		virtual void accept(const symel_cycleperm<N, T> &elem)
			throw(exception) {
			m_eq = m_elem.equals(elem);
		}
		bool get_equals() { return m_eq; }
	};

private:
	size_t m_ord; //!< Cycle order
	mask<N> m_msk; //!< Mask of affected indexes
	transf<N, T> m_tr; //!< Transformation

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates the cyclic %permutation element
		\param ord Cycle order (length), must be at least 2 and not
			exceed the maximum length specified in the mask.
		\param msk Mask of affected indexes.
		\throw bad_parameter If the %mask is invalid.
		\throw out_of_bounds If the cycle order is out of bounds.
	 **/
	symel_cycleperm(size_t ord, const mask<N> &msk);

	/**	\brief Creates the cyclic %permutation element
		\param ord Cycle order (length), must be at least 2 and not
			exceed the maximum length specified in the mask.
		\param msk Mask of affected indexes.
		\param tr Transformation that specifies the sign function.
		\throw bad_parameter If the %mask or the transformation are
			invalid.
		\throw out_of_bounds If the cycle order is out of bounds.
	 **/
	symel_cycleperm(size_t ord, const mask<N> &msk, const transf<N, T> &tr);

	/**	\brief Copy constructor
		\param elem Source cyclic %permutation object.
	 **/
	symel_cycleperm(const symel_cycleperm<N, T> &elem);

	/**	\brief Virtual destructor
	 **/
	virtual ~symel_cycleperm();

	//@}

	//!	\name Implementation of symmetry_element_i<N, T>
	//@{
	virtual const mask<N> &get_mask() const;
	virtual void permute(const permutation<N> &perm);
	virtual bool is_valid_bis(const block_index_space<N> &bis) const;
	virtual bool is_allowed(const index<N> &idx) const;
	virtual void apply(index<N> &idx) const;
	virtual void apply(index<N> &idx, transf<N, T> &tr) const;
	virtual bool equals(const symmetry_element_i<N, T> &se) const;
	virtual symmetry_element_i<N, T> *clone() const;
	//@}

	bool equals(const symel_cycleperm<N, T> &elem) const;
//	const permutation<N> &get_perm() const;
	size_t get_order() const;
	const transf<N, T> &get_transf() const;

private:
	void build_cycle() throw(bad_parameter, out_of_bounds);
};



template<size_t N, typename T>
const char *symel_cycleperm<N, T>::k_clazz = "symel_cycleperm<N, T>";


template<size_t N, typename T>
symel_cycleperm<N, T>::symel_cycleperm(size_t ord, const mask<N> &msk)
: m_ord(ord), m_msk(msk) {

	build_cycle();
}


template<size_t N, typename T>
symel_cycleperm<N, T>::symel_cycleperm(
	size_t ord, const mask<N> &msk, const transf<N, T> &tr)
: m_ord(ord), m_msk(msk), m_tr(tr) {

	build_cycle();
}


template<size_t N, typename T>
symel_cycleperm<N, T>::symel_cycleperm(const symel_cycleperm<N, T> &elem)
: m_ord(elem.m_ord), m_msk(elem.m_msk), m_tr(elem.m_tr) {

}


template<size_t N, typename T>
symel_cycleperm<N, T>::~symel_cycleperm() {

}


template<size_t N, typename T>
const mask<N> &symel_cycleperm<N, T>::get_mask() const {

	return m_msk;
}


template<size_t N, typename T>
void symel_cycleperm<N, T>::permute(const permutation<N> &perm) {

	m_msk.permute(perm);
	m_tr.permute(perm);
}


template<size_t N, typename T>
bool symel_cycleperm<N, T>::is_valid_bis(
	const block_index_space<N> &bis) const {

	bool first = true;
	size_t type;
	for(register size_t i = 0; i < N; i++) {
		if(m_msk[i]) {
			if(first) {
				type = bis.get_type(i);
				first = false;
			} else {
				if(bis.get_type(i) != type) return false;
			}
		}
	}
	return true;
}


template<size_t N, typename T>
bool symel_cycleperm<N, T>::is_allowed(const index<N> &idx) const {

	return true;
}



template<size_t N, typename T>
void symel_cycleperm<N, T>::apply(index<N> &idx) const {

	m_tr.apply(idx);
}


template<size_t N, typename T>
void symel_cycleperm<N, T>::apply(index<N> &idx, transf<N, T> &tr) const {

	m_tr.apply(idx);
	tr.transform(m_tr);
}


template<size_t N, typename T>
bool symel_cycleperm<N, T>::equals(const symmetry_element_i<N, T> &se) const {

	equals_functor eq(*this);
	se.dispatch(eq);
	return eq.get_equals();
}


template<size_t N, typename T>
symmetry_element_i<N, T> *symel_cycleperm<N, T>::clone() const {

	return new symel_cycleperm<N, T>(*this);
}


template<size_t N, typename T>
bool symel_cycleperm<N, T>::equals(const symel_cycleperm<N, T> &elem) const {

	if(this == &elem) return true;
	return m_ord == elem.m_ord && m_msk.equals(elem.m_msk);
}


template<size_t N, typename T>
inline size_t symel_cycleperm<N, T>::get_order() const {

	return m_ord;
}


template<size_t N, typename T>
inline const transf<N, T> &symel_cycleperm<N, T>::get_transf() const {

	return m_tr;
}

//template<size_t N, typename T>
//inline const permutation<N> &symel_cycleperm<N, T>::get_perm() const {
//
//	return m_perm;
//}


template<size_t N, typename T>
void symel_cycleperm<N, T>::build_cycle() throw(bad_parameter, out_of_bounds) {

	static const char *method = "build_cycle()";

	size_t nset = 0;
	for(register size_t i = 0; i < N; i++) if(m_msk[i]) nset++;
	if(nset < 2) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Invalid mask.");
	}
	if(m_ord < 2 || m_ord > nset) {
		throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Cycle order is out of bounds.");
	}

	m_tr.get_perm().reset();

	size_t i1 = 0, i2, len = 1;
	while(i1 < N && m_msk[i1] == false) i1++;
	while(i1 < N) {
		i2 = i1 + 1;
		while(i2 < N && m_msk[i2] == false) i2++;
		if(i2 < N && len < m_ord) {
			m_tr.get_perm().permute(i1, i2);
			len++;
		}
		i1 = i2;
	}

	transf<N, T> tr2;
	for(size_t i = 0; i < m_ord; i++) tr2.transform(m_tr);
	if(!tr2.is_identity()) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Transformation yields an inconsistent cycle.");
	}
}

} // namespace libtensor

#endif // LIBTENSOR_SYMEL_CYCLEPERM_H
