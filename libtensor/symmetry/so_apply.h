#ifndef LIBTENSOR_SO_APPLY_H
#define LIBTENSOR_SO_APPLY_H

#include <libtensor/core/symmetry.h>
#include "symmetry_operation_base.h"
#include "symmetry_operation_params.h"

namespace libtensor {

template<size_t N, typename T>
class so_apply;

template<size_t N, typename T>
class symmetry_operation_params< so_apply<N, T> >;


/** \brief Computes the %symmetry of a tensor subjected to some functor
    \tparam N Symmetry cardinality (%tensor order).
    \tparam T Tensor element type.

    The symmetry operation computes the symmetry of a tensor \f$ T \f$ whose
    elements have been subjected to a function \f$ f(x) \f$:
    \f[
        T'_{ij...} = f\left(T_{ij...}\right)
    \f]
    To perform this task the operation requires two types of information
    about the function:
    - if the function maps 0 onto 0 (i.e. if elements which were zero due to
      symmetry stay zero
    - if a scalar transformation \f$ \hat{S} \f$ on the tensor elements
      results in a scalar transformation \f$ \hat{S}' \f$ of the elements of
      the result tensor (i.e. if
      \f$ f\left(\hat{S} x\right) = \hat{S}' f(x) \f$).

    If the scalar transformation \f$ \hat{S} \f$ is the identity transformation
    the function is assumed to be asymmetric.


    \ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_apply : public symmetry_operation_base< so_apply<N, T> > {
private:
    typedef so_apply<N, T> operation_t;
    typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

private:
    const symmetry<N, T> &m_sym1; //!< Symmetry container (A)
    permutation<N> m_perm1; //!< Permutation of the %tensor
    scalar_transf<T> m_s1; //!< Scalar transformation \f$ \hat{S} \f$
    scalar_transf<T> m_s2; //!< Scalar transformation \f$ \hat{S}' \f$
    bool m_keep_zero; //!< Functor maps 0 to 0

public:
    /** \brief Initializes the operation
        \param sym %Symmetry container (A).
        \param perm Permutation of the %tensor.
        \param is_asym Functor is asymmetric.
        \param sign Functor is symmetric or anti-symmetric
            (ignored if is_asym is true).
     **/
    so_apply(const symmetry<N, T> &sym1, const permutation<N> &perm1,
		const scalar_transf<T> &s1, const scalar_transf<T> &s2,
		bool keep_zero) : m_sym1(sym1), m_perm1(perm1),
		m_s1(s1), m_s2(s2), m_keep_zero(keep_zero) {

    }

    /** \brief Performs the operation
        \param sym Destination %symmetry container.
     **/
    void perform(symmetry<N, T> &sym);

private:
    so_apply(const so_apply<N, T>&);
    const so_apply<N, T> &operator=(const so_apply<N, T>&);
};

template<size_t N, typename T>
class symmetry_operation_params< so_apply<N, T> > :
    public symmetry_operation_params_i {

public:
    const symmetry_element_set<N, T> &grp1; //!< Symmetry group 1
    permutation<N> perm1; //!< Permutation 1
    scalar_transf<T> s1;
    scalar_transf<T> s2;
    bool keep_zero; //!< Functor maps 0 to 0
    symmetry_element_set<N, T> &grp2; //!< Symmetry group 2 (output)

public:
    symmetry_operation_params(
            const symmetry_element_set<N, T> &grp1_,
            const permutation<N> &perm1_,
            const scalar_transf<T> &s1_,
            const scalar_transf<T> &s2_,
            bool keep_zero_,
            symmetry_element_set<N, T> &grp2_) :

                grp1(grp1_), perm1(perm1_), s1(s1_), s2(s2_),
                keep_zero(keep_zero_), grp2(grp2_) { }

    virtual ~symmetry_operation_params() { }
};


} // namespace libtensor

#include "so_apply_handlers.h"

#endif // LIBTENSOR_SO_APPLY_H

