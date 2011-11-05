#ifndef LIBTENSOR_SO_REDUCE_H
#define LIBTENSOR_SO_REDUCE_H

#include "../core/mask.h"
#include "../core/symmetry.h"
#include "../core/symmetry_element_set.h"
#include "bad_symmetry.h"
#include "so_copy.h"
#include "symmetry_operation_base.h"
#include "symmetry_operation_dispatcher.h"
#include "symmetry_operation_params.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename T>
class so_reduce;

template<size_t N, size_t M, size_t K, typename T>
class symmetry_operation_params< so_reduce<N, M, K, T> >;


/**	\brief Projection of a %symmetry group onto a subspace
	\tparam N Order of the argument space.
	\tparam M Decrement in the order of the result space.
	\tparam K Number of separate stabilizations to perform.

	The operation takes a %symmetry group that is defined for a %tensor
	space of order N and produces a group that acts in a %tensor space
	of order N - M by doing K separate reductions.
    The masks specify the dimensions which are to be reduced, i.e. which do not
	remain in the result.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, size_t K, typename T>
class so_reduce : public symmetry_operation_base< so_reduce<N, M, K, T> > {
private:
    typedef so_reduce<N, M, K, T> operation_t;
    typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;
public:
    static const char *k_clazz;

private:
    const symmetry<N, T> &m_sym1;
    mask<N> m_msk[K];
    size_t m_msk_set;
public:
    so_reduce(const symmetry<N, T> &sym1) : m_sym1(sym1), m_msk_set(0) {}

    void add_mask(const mask<N> &msk) {
        m_msk[m_msk_set++] = msk;
    }

    void perform(symmetry<N - M, T> &sym2);

};

template<size_t N, size_t M, size_t K, typename T>
const char *so_reduce<N, M, K, T>::k_clazz = "so_reduce<N, M, K, T>";

/**	\brief Projection of a %symmetry group onto vacuum (specialization)
	\tparam N Order.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t K, typename T>
class so_reduce<N, N, K, T> {
public:
    so_reduce(const symmetry<N, T> &sym1) { }

    void add_mask(const mask<N> &msk) { }

    void perform(symmetry<0, T> &sym2) {
        sym2.clear();
    }
};


/**	\brief Projection of a %symmetry group onto itself (specialization)
	\tparam N Order.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t K, typename T>
class so_reduce<N, 0, K, T> {
private:
    const symmetry<N, T> &m_sym1;
public:
    so_reduce(const symmetry<N, T> &sym1) :
        m_sym1(sym1) { }

    void add_mask(const mask<N> &msk) { }

    void perform(symmetry<N, T> &sym2) {
        so_copy<N, T>(m_sym1).perform(sym2);
    }
};


template<size_t N, size_t M, size_t K, typename T>
void so_reduce<N, M, K, T>::perform(symmetry<N - M, T> &sym2) {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "perform(symmetry<N - M, T> &)";
    if (m_msk_set != K) {
        bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Masks not set properly.");
    }
#endif

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


template<size_t N, size_t M, size_t K, typename T>
class symmetry_operation_params< so_reduce<N, M, K, T> > :
public symmetry_operation_params_i {

public:
    const symmetry_element_set<N, T> &grp1; //!< Symmetry group
    mask<N> msk[K]; //!< Mask
    symmetry_element_set<N - M, T> &grp2;

public:
    symmetry_operation_params(
            const symmetry_element_set<N, T> &grp1_,
            const mask<N> (&msk_)[K],
            symmetry_element_set<N - M, T> &grp2_) :

                grp1(grp1_), grp2(grp2_) {

        for(size_t i = 0; i < K; i++) msk[i] = msk_[i];
    }

    virtual ~symmetry_operation_params() { }
};


} // namespace libtensor

#include "so_reduce_handlers.h"

#endif // LIBTENSOR_SO_REDUCE_H

