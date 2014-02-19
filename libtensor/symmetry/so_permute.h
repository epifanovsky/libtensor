#ifndef LIBTENSOR_SO_PERMUTE_H
#define LIBTENSOR_SO_PERMUTE_H

#include "../core/symmetry.h"
#include "symmetry_operation_base.h"
#include "symmetry_operation_params.h"

namespace libtensor {


template<size_t N, typename T>
class so_permute;

template<size_t N, typename T>
class symmetry_operation_params< so_permute<N, T> >;


/** \brief Adjusts %symmetry elements to a %permutation of %tensor indexes
    \tparam N Symmetry cardinality (%tensor order).
    \tparam T Tensor element type.

    \ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_permute : public symmetry_operation_base< so_permute<N, T> > {
private:
    typedef so_permute<N, T> operation_t;
    typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

private:
    const symmetry<N, T> &m_sym1; //!< Symmetry container
    permutation<N> m_perm; //!< Permutation

public:
    /** \brief Initializes the operation
        \param sym1 Source %symmetry container.
        \param perm Permutation of %tensor indexes.
     **/
    so_permute(const symmetry<N, T> &sym1, const permutation<N> &perm) :
        m_sym1(sym1), m_perm(perm) { }

    /** \brief Performs the operation
        \param sym2 Destination %symmetry container.
     **/
    void perform(symmetry<N, T> &sym2);

private:
    so_permute(const so_permute<N, T>&);
    const so_permute<N, T> &operator=(const so_permute<N, T>&);
};

template<size_t N, typename T>
class symmetry_operation_params< so_permute<N, T> > :
public symmetry_operation_params_i {

public:
    const symmetry_element_set<N, T> &g1; //!< Symmetry group 1
    permutation<N> perm; //!< Permutation
    symmetry_element_set<N, T> &g2; //!< Symmetry group 2

public:
    symmetry_operation_params(
            const symmetry_element_set<N, T> &g1_,
            const permutation<N> &perm_,
            symmetry_element_set<N, T> &g2_) :

                g1(g1_), perm(perm_), g2(g2_) { }

    virtual ~symmetry_operation_params() { }
};

} // namespace libtensor

#include "so_permute_handlers.h"

#endif // LIBTENSOR_SO_PERMUTE_H

