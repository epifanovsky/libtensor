#ifndef LIBTENSOR_SO_DIRPROD_H
#define LIBTENSOR_SO_DIRPROD_H

#include "../core/symmetry.h"
#include "../core/symmetry_element_set.h"
#include "so_permute.h"
#include "symmetry_operation_base.h"
#include "symmetry_operation_dispatcher.h"
#include "symmetry_operation_params.h"

namespace libtensor {


template<size_t N, size_t M, typename T>
class so_dirprod;

template<size_t N, size_t M, typename T>
class symmetry_operation_params< so_dirprod<N, M, T> >;


/**	\brief Direct product of two %symmetry groups
	\tparam N Order of the first %symmetry group.
	\tparam M Order of the second %symmetry group.

	The operation forms the direct product of two given %symmetry groups.

	The operation takes two %symmetry group that are defined for %tensor
	spaces of order N and M, respectively and produces a group that acts in
	a %tensor space of order N + M.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class so_dirprod : public symmetry_operation_base< so_dirprod<N, M, T> > {
private:
    typedef so_dirprod<N, M, T> operation_t;
    typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

private:
    const symmetry<N, T> &m_sym1;
    const symmetry<M, T> &m_sym2;
    permutation<N + M> m_perm;

public:
    so_dirprod(const symmetry<N, T> &sym1, const symmetry<M, T> &sym2,
            const permutation<N + M> &perm) :
                m_sym1(sym1), m_sym2(sym2), m_perm(perm)
    { }

    so_dirprod(const symmetry<N, T> &sym1, const symmetry<M, T> &sym2) :
        m_sym1(sym1), m_sym2(sym2)
    { }

    void perform(symmetry<N + M, T> &sym3);

private:
    template<size_t X>
    void copy_subset(const symmetry_element_set<X, T> &set1,
            symmetry<X, T> &sym2);
};


/**	\brief Direct product of vacuum with other symmetry (specialization)
	\tparam M Order.

	\ingroup libtensor_symmetry
 **/
template<size_t M, typename T>
class so_dirprod<0, M, T> {
private:
    const symmetry<M, T> &m_sym2;
    permutation<M> m_perm;

public:
    so_dirprod(const symmetry<0, T> &sym1, const symmetry<M, T> &sym2) :
        m_sym2(sym2)
    { }

    so_dirprod(const symmetry<0, T> &sym1, const symmetry<M, T> &sym2,
            const permutation<M> &perm) : m_sym2(sym2), m_perm(perm)
    { }

    void perform(symmetry<M, T> &sym3) {

        sym3.clear();
        so_permute<M, T>(m_sym2, m_perm).perform(sym3);
    }
};

/**	\brief Direct product of %symmetry group with vacuum (specialization)
	\tparam N Order.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_dirprod<N, 0, T> {
private:
    const symmetry<N, T> &m_sym1;
    permutation<N> m_perm;

public:
    so_dirprod(const symmetry<N, T> &sym1, const symmetry<0, T> &sym2) :
        m_sym1(sym1)
    { }

    so_dirprod(const symmetry<N, T> &sym1, const symmetry<0, T> &sym2,
            const permutation<N> &perm) : m_sym1(sym1), m_perm(perm)
    { }

    void perform(symmetry<N, T> &sym3) {

        sym3.clear();
        so_permute<N, T>(m_sym1, m_perm).perform(sym3);
    }
};

/** \brief Parameters for symmetry operation so_dirprod
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_params< so_dirprod<N, M, T> > :
public symmetry_operation_params_i {

public:
    const symmetry_element_set<N, T> &g1; //!< Symmetry group 1
    const symmetry_element_set<M, T> &g2; //!< Symmetry group 2
    permutation<N + M> perm; //!< Permutation
    block_index_space<N + M> bis; //!< Block index space of result
    symmetry_element_set<N + M, T> &g3; //!< Result symmetry group

public:
    symmetry_operation_params(
            const symmetry_element_set<N, T> &g1_,
            const symmetry_element_set<M, T> &g2_,
            const permutation<N + M> &perm_,
            const block_index_space<N + M> &bis_,
            symmetry_element_set<N + M, T> &g3_) :
                g1(g1_), g2(g2_), perm(perm_), bis(bis_), g3(g3_)
    { }

};


} // namespace libtensor

#include "so_dirprod_handlers.h"

#endif // LIBTENSOR_SO_DIRPROD_H
