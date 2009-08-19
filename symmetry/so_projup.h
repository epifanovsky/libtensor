#ifndef LIBTENSOR_SO_PROJUP_H
#define LIBTENSOR_SO_PROJUP_H

#include "defs.h"
#include "exception.h"
#include "core/mask.h"
#include "core/dimensions.h"
#include "core/symmetry_element_i.h"
#include "symel_cycleperm.h"
#include "symmetry_element_target.h"

namespace libtensor {


/**	\brief Projects a %symmetry element onto a larger space

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class so_projup {
private:
	class proj :
		public symmetry_element_target< N, T, symel_cycleperm<N, T> > {
	private:
		mask<N + M> m_msk;
		dimensions<N + M> m_dims;
		symmetry_element_i<N + M, T> *m_elem;

	public:
		proj(const mask<N + M> &msk, const dimensions<N + M> &dims);
		virtual void accept(const symel_cycleperm<N, T> &elem)
			throw(exception);
		const symmetry_element_i<N + M, T> &get_proj() const;
	};

private:
	proj m_proj;

public:
	so_projup(const symmetry_element_i<N, T> &elem,
		const mask<N + M> &msk, const dimensions<N + M> &dims);
	const symmetry_element_i<N + M, T> &get_proj() const;
};


template<size_t N, size_t M, typename T>
so_projup<N, M, T>::so_projup(const symmetry_element_i<N, T> &elem,
	const mask<N + M> &msk, const dimensions<N + M> &dims)
: m_proj(msk, dims) {

	elem.dispatch(m_proj);
}


template<size_t N, size_t M, typename T>
const symmetry_element_i<N + M, T> &so_projup<N, M, T>::get_proj() const {

	return m_proj.get_proj();
}


template<size_t N, size_t M, typename T>
so_projup<N, M, T>::proj::proj(
	const mask<N + M> &msk, const dimensions<N + M> &dims)
: m_msk(msk), m_dims(dims), m_elem(NULL) {

	register size_t m = 0;
	for(register size_t i = 0; i < N + M; i++) {
		if(m_msk[i]) m++;
	}
	if(m != N) {
		throw_exc("so_projup<N, M, T>::proj",
			"proj(const mask<N + M>&)", "Invalid mask.");
	}
}


template<size_t N, size_t M, typename T>
void so_projup<N, M, T>::proj::accept(const symel_cycleperm<N, T> &elem)
	throw(exception) {

	const mask<N> &oldmask = elem.get_mask();
	mask<N + M> newmask;
	for(register size_t i = 0, j = 0; i < N + M; i++) {
		if(m_msk[i]) {
			newmask[i] = oldmask[j];
			j++;
		} else {
			newmask[i] = false;
		}
	}
	if(m_elem) delete m_elem;
	m_elem = new symel_cycleperm<N + M, T>(newmask, m_dims);
}


template<size_t N, size_t M, typename T>
const symmetry_element_i<N + M, T>&so_projup<N, M, T>::proj::get_proj() const {

	if(m_elem == NULL) {
		throw_exc("so_projup<N, M, T>::proj", "get_proj()",
			"NULL pointer.");
	}
	return *m_elem;
}


} // namespace libtensor

#endif // LIBTENSOR_SO_PROJUP_H
