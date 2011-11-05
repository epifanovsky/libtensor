#ifndef LIBTENSOR_SO_PROJ_DOWN_H
#define LIBTENSOR_SO_PROJ_DOWN_H

#include "../core/mask.h"
#include "../core/symmetry.h"
#include "../core/symmetry_element_set.h"
#include "symmetry_operation_base.h"
#include "symmetry_operation_dispatcher.h"
#include "symmetry_operation_params.h"

namespace libtensor {


template<size_t N, size_t M, typename T>
class so_proj_down;

template<size_t N, size_t M, typename T>
class symmetry_operation_params< so_proj_down<N, M, T> >;


/**	\brief Projection of a %symmetry group onto a subspace
	\tparam N Order of the argument space.
	\tparam M Decrement in the order of the result space.

	The operation takes a %symmetry group that is defined for a %tensor
	space of order N and produces a group that acts in a %tensor space
	of order N - M.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class so_proj_down : public symmetry_operation_base< so_proj_down<N, M, T> > {
private:
    typedef so_proj_down<N, M, T> operation_t;
    typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

private:
    const symmetry<N, T> &m_sym1;
    mask<N> m_msk;

public:
    so_proj_down(const symmetry<N, T> &sym1, const mask<N> &msk) :
        m_sym1(sym1), m_msk(msk) { }

    void perform(symmetry<N - M, T> &sym2);

};


/**	\brief Projection of a %symmetry group onto vacuum (specialization)
	\tparam N Order.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_proj_down<N, N, T> {
public:
    so_proj_down(const symmetry<N, T> &sym1, const mask<N> &msk) { }

    void perform(symmetry<0, T> &sym2) {
        sym2.clear();
    }
};


template<size_t N, size_t M, typename T>
void so_proj_down<N, M, T>::perform(symmetry<N - M, T> &sym2) {

    for(typename symmetry<N, T>::iterator i = m_sym1.begin();
            i != m_sym1.end(); i++) {

        const symmetry_element_set<N, T> &set1 =
                m_sym1.get_subset(i);
        symmetry_element_set<N - M, T> set2(set1.get_id());
        symmetry_operation_params<operation_t> params(
                set1, m_msk, set2);
        dispatcher_t::get_instance().invoke(set1.get_id(), params);

        for(typename symmetry_element_set<N - M, T>::iterator j =
                set2.begin(); j != set2.end(); j++) {
            sym2.insert(set2.get_elem(j));
        }
    }
}


template<size_t N, size_t M, typename T>
class symmetry_operation_params< so_proj_down<N, M, T> > :
public symmetry_operation_params_i {

public:
    const symmetry_element_set<N, T> &grp1; //!< Symmetry group
    mask<N> msk; //!< Mask
    symmetry_element_set<N - M, T> &grp2;

public:
    symmetry_operation_params(
            const symmetry_element_set<N, T> &grp1_,
            const mask<N> &msk_,
            symmetry_element_set<N - M, T> &grp2_) :

                grp1(grp1_), msk(msk_), grp2(grp2_) { }

    virtual ~symmetry_operation_params() { }
};


} // namespace libtensor

#include "so_proj_down_handlers.h"

#endif // LIBTENSOR_SO_PROJ_DOWN_H

